"""Sleep BRL EDF dataset support."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .base import LABEL_MAP, build_sequences, scale_epochs_channelwise
from .capslpdb import _read_capslpdb_signals

DEFAULT_SLEEPBRL_FEATURES: Sequence[str] = tuple(f"S{i}" for i in range(1, 17))

# The Sleep BRL annotations encode stages as ASCII digits.
SLEEPBRL_STAGE_MAP: Dict[str, int] = {
    "1": LABEL_MAP["W"],
    "2": LABEL_MAP["N2"],
    "3": LABEL_MAP["R"],
}


def _parse_sleepbrl_annotations(
    annotation_path: Path,
    total_samples: int,
    sample_rate_hz: float,
    epoch_seconds: int,
) -> List[int]:
    """Parse WFDB-style annotation file exported as .edf.atr."""
    data = np.fromfile(annotation_path, dtype="<u2")
    sample_index = 0
    events: List[Tuple[int, int]] = []

    for word in data:
        ann_type = int(word & 0x3F)
        sample_index += int(word >> 6)
        if 32 <= ann_type <= 126:  # printable ASCII
            stage_char = chr(ann_type)
            if stage_char in SLEEPBRL_STAGE_MAP:
                events.append((sample_index, SLEEPBRL_STAGE_MAP[stage_char]))

    if not events:
        raise ValueError(f"No recognizable sleep stage events found in {annotation_path.name}")

    events.sort(key=lambda item: item[0])

    samples_per_epoch = int(sample_rate_hz * epoch_seconds)
    num_epochs = total_samples // samples_per_epoch
    epoch_labels: List[int] = []
    event_idx = 0
    current_label = events[0][1]

    for epoch in range(num_epochs):
        epoch_start_sample = epoch * samples_per_epoch
        while event_idx + 1 < len(events) and events[event_idx + 1][0] <= epoch_start_sample:
            event_idx += 1
            current_label = events[event_idx][1]
        epoch_labels.append(current_label)

    return epoch_labels


class SleepBrlDataset(Dataset):
    """Dataset that loads Sleep BRL EDF pairs (.edf + .edf.atr)."""

    def __init__(
        self,
        edf_files: Sequence[str],
        sequence_length: int = 10,
        epoch_seconds: int = 30,
        sample_rate_hz: int = 50,
        feature_columns: Optional[Sequence[str]] = None,
    ):
        self.edf_files = list(edf_files)
        self.sequence_length = sequence_length
        self.epoch_seconds = epoch_seconds
        self.sample_rate_hz = sample_rate_hz
        self.samples_per_epoch = int(self.sample_rate_hz * self.epoch_seconds)
        self.feature_columns = list(feature_columns or DEFAULT_SLEEPBRL_FEATURES)

        self.sequences, self.labels = self._load_and_preprocess_data()
        if self.sequences.size == 0:
            raise ValueError(
                "No valid sequences were created from Sleep BRL files. "
                "Verify that the EDF files have matching .edf.atr annotations "
                "and that the requested channels exist."
            )

        self.num_channels = self.sequences.shape[2]
        self.feature_dim = self.sequences.shape[2] * self.sequences.shape[3]

    def _find_annotation(self, edf_path: Path) -> Optional[Path]:
        candidate = edf_path.with_suffix(".edf.atr")
        return candidate if candidate.exists() else None

    def _load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        all_sequences: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        iterator: Iterable[str]
        if len(self.edf_files) > 1:
            iterator = tqdm(self.edf_files, desc="preprocess-sleepbrl", unit="file")
        else:
            iterator = self.edf_files

        for file_path_str in iterator:
            file_path = Path(file_path_str)
            annotation_path = self._find_annotation(file_path)
            if annotation_path is None:
                print(f"No annotation (.edf.atr) found for {file_path.name}. Skipping.", flush=True)
                continue

            print(f"\n--- Processing SleepBRL record: {file_path.name} ---", flush=True)
            try:
                signals, sample_rates, duration = _read_capslpdb_signals(file_path, self.feature_columns)
                target_rate = float(self.sample_rate_hz)
                if not np.isclose(sample_rates[self.feature_columns[0]], target_rate):
                    # simple linear interpolation resampling
                    total_target_samples = int(round(duration * target_rate))
                    src_times = np.linspace(0, duration, num=signals.shape[1], endpoint=False)
                    tgt_times = np.linspace(0, duration, num=total_target_samples, endpoint=False)
                    signals = np.vstack(
                        [np.interp(tgt_times, src_times, channel) for channel in signals]
                    ).astype(np.float32)
                total_samples = signals.shape[1]
                epoch_labels = _parse_sleepbrl_annotations(
                    annotation_path, total_samples, target_rate, self.epoch_seconds
                )
                samples_per_epoch = int(target_rate * self.epoch_seconds)
                num_epochs_available = min(len(epoch_labels), total_samples // samples_per_epoch)
                epochs = []
                labels = []
                for idx in range(num_epochs_available):
                    start = idx * samples_per_epoch
                    end = start + samples_per_epoch
                    epochs.append(signals[:, start:end])
                    labels.append(epoch_labels[idx])
                if not epochs:
                    continue
                epochs_array = np.stack(epochs)
                labels_array = np.asarray(labels, dtype=np.int64)
            except Exception as err:
                print(f"Failed to process {file_path.name}: {err}", flush=True)
                continue

            scaled_epochs = scale_epochs_channelwise(epochs_array)
            sequences, seq_labels = build_sequences(scaled_epochs, labels_array, self.sequence_length)
            if sequences.size == 0:
                continue
            all_sequences.append(sequences)
            all_labels.append(seq_labels)

        if not all_sequences:
            return (
                np.empty((0, self.sequence_length, len(self.feature_columns), self.samples_per_epoch), dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )

        return np.concatenate(all_sequences, axis=0), np.concatenate(all_labels, axis=0)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.sequences[idx]).to(torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
