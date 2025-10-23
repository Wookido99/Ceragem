"""CAPSLPDB EDF dataset support."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .base import LABEL_MAP, build_sequences, canonical_channel, scale_epochs_channelwise

DEFAULT_CAP_FEATURE_COLUMNS: Sequence[str] = (
    "Fp2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "F8-T4",
    "T4-T6",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "F7-T3",
    "T3-T5",
    "C4-A1",
    "ROC-LOC",
    "EMG1-EMG2",
    "ECG1-ECG2",
    "DX1-DX2",
    "SX1-SX2",
)

CAP_STAGE_TO_LABEL: Dict[str, Optional[int]] = {
    "SLEEP-S0": LABEL_MAP["W"],
    "SLEEP-S1": LABEL_MAP["N1"],
    "SLEEP-S2": LABEL_MAP["N2"],
    "SLEEP-S3": LABEL_MAP["N3"],
    "SLEEP-S4": LABEL_MAP["N3"],  # Merge stages 3 & 4 into N3
    "SLEEP-REM": LABEL_MAP["R"],
    "SLEEP-MT": None,
    "SLEEP-UNSCORED": None,
}


def _parse_capslpdb_stage_file(annotation_path: Path, epoch_seconds: int) -> List[int]:
    with open(annotation_path, "r", errors="ignore") as handle:
        text = handle.read()

    import re

    matches = re.finditer(r"(SLEEP-[A-Z0-9]+)\s+(\d+)", text)
    epoch_labels: List[int] = []
    for match in matches:
        desc = match.group(1)
        duration = int(match.group(2))
        label = CAP_STAGE_TO_LABEL.get(desc)
        if label is None:
            continue
        epochs = max(1, int(round(duration / float(epoch_seconds))))
        epoch_labels.extend([label] * epochs)
    return epoch_labels


def _read_capslpdb_signals(
    edf_path: Path,
    feature_columns: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, float], float]:
    feature_columns = list(feature_columns)
    canonical_features = [canonical_channel(ch) for ch in feature_columns]

    with open(edf_path, "rb") as handle:
        header = handle.read(256)
        num_records_raw = header[236:244].decode("ascii", errors="ignore").strip()
        try:
            num_records_int = int(num_records_raw)
        except ValueError:
            num_records_int = -1
        record_duration = float(header[244:252].decode("ascii", errors="ignore").strip() or "1.0")
        num_signals = int(header[252:256].decode("ascii", errors="ignore").strip())

        label_bytes = handle.read(16 * num_signals)
        labels = [
            label_bytes[i * 16 : (i + 1) * 16].decode("ascii", errors="ignore").strip()
            for i in range(num_signals)
        ]

        handle.seek(80 * num_signals, os.SEEK_CUR)  # transducer type
        handle.seek(8 * num_signals, os.SEEK_CUR)  # physical dimension

        phys_min_bytes = handle.read(8 * num_signals)
        phys_max_bytes = handle.read(8 * num_signals)
        dig_min_bytes = handle.read(8 * num_signals)
        dig_max_bytes = handle.read(8 * num_signals)

        handle.seek(80 * num_signals, os.SEEK_CUR)  # prefiltering
        samples_per_record_bytes = handle.read(8 * num_signals)
        handle.seek(32 * num_signals, os.SEEK_CUR)  # reserved

        phys_min = np.array(
            [
                float(phys_min_bytes[i * 8 : (i + 1) * 8].decode("ascii", errors="ignore").strip() or "0")
                for i in range(num_signals)
            ],
            dtype=np.float64,
        )
        phys_max = np.array(
            [
                float(phys_max_bytes[i * 8 : (i + 1) * 8].decode("ascii", errors="ignore").strip() or "1")
                for i in range(num_signals)
            ],
            dtype=np.float64,
        )
        dig_min = np.array(
            [
                float(dig_min_bytes[i * 8 : (i + 1) * 8].decode("ascii", errors="ignore").strip() or "-32768")
                for i in range(num_signals)
            ],
            dtype=np.float64,
        )
        dig_max = np.array(
            [
                float(dig_max_bytes[i * 8 : (i + 1) * 8].decode("ascii", errors="ignore").strip() or "32767")
                for i in range(num_signals)
            ],
            dtype=np.float64,
        )
        samples_per_record = np.array(
            [
                int(float(samples_per_record_bytes[i * 8 : (i + 1) * 8].decode("ascii", errors="ignore").strip() or "0"))
                for i in range(num_signals)
            ],
            dtype=np.int32,
        )

        header_bytes = 256 + num_signals * 256
        record_bytes = int(np.sum(samples_per_record) * 2)
        file_size = os.path.getsize(edf_path)
        actual_records = (file_size - header_bytes) // record_bytes if record_bytes > 0 else 0
        if num_records_int <= 0 or (actual_records and num_records_int > actual_records):
            num_records_int = actual_records

        scale = (phys_max - phys_min) / (dig_max - dig_min)
        offset = phys_max - scale * dig_max

        label_lookup: Dict[str, int] = {}
        for idx, label in enumerate(labels):
            canonical = canonical_channel(label)
            label_lookup[canonical] = idx

        channel_indices: List[int] = []
        for canon_feature in canonical_features:
            if canon_feature in label_lookup:
                channel_indices.append(label_lookup[canon_feature])
                continue
            matches = [
                (idx, lab)
                for idx, lab in enumerate(labels)
                if canonical_channel(lab).endswith(canon_feature)
            ]
            if matches:
                channel_indices.append(matches[0][0])
            else:
                raise ValueError(
                    f"Channel '{feature_columns[len(channel_indices)]}' not present in EDF file '{edf_path.name}'."
                )

        total_samples = {
            idx: samples_per_record[idx] * num_records_int for idx in channel_indices
        }
        buffers = {
            idx: np.empty(total_samples[idx], dtype=np.float32) for idx in channel_indices
        }
        write_pos = {idx: 0 for idx in channel_indices}

        for _ in range(num_records_int):
            for sig_idx in range(num_signals):
                samples = samples_per_record[sig_idx]
                block = handle.read(samples * 2)
                if len(block) < samples * 2:
                    samples = len(block) // 2
                    block = block[: samples * 2]
                if samples == 0 or sig_idx not in buffers:
                    continue
                raw_vals = np.frombuffer(block, dtype="<i2").astype(np.float64)
                transformed = raw_vals * scale[sig_idx] + offset[sig_idx]
                dest = buffers[sig_idx]
                pos = write_pos[sig_idx]
                dest[pos : pos + samples] = transformed.astype(np.float32)
                write_pos[sig_idx] += samples

    for idx in channel_indices:
        buffers[idx] = buffers[idx][: write_pos[idx]]

    signals = np.vstack([buffers[idx] for idx in channel_indices])
    sample_rates = {
        feature_columns[i]: samples_per_record[channel_indices[i]] / record_duration
        for i in range(len(channel_indices))
    }
    total_duration = num_records_int * record_duration
    return signals, sample_rates, total_duration


def _load_capslpdb_record(
    edf_path: Path,
    annotation_path: Path,
    feature_columns: Sequence[str],
    sample_rate_hz: int,
    epoch_seconds: int,
) -> Tuple[np.ndarray, np.ndarray]:
    raw_signals, sample_rates, total_duration = _read_capslpdb_signals(edf_path, feature_columns)
    target_rate = float(sample_rate_hz)
    target_samples = int(round(total_duration * target_rate))

    def _resample(channel_data: np.ndarray, current_rate: float) -> np.ndarray:
        if np.isclose(current_rate, target_rate):
            return channel_data.astype(np.float32, copy=False)
        src_times = np.linspace(0, total_duration, num=channel_data.shape[-1], endpoint=False)
        tgt_times = np.linspace(0, total_duration, num=target_samples, endpoint=False)
        resampled = np.interp(tgt_times, src_times, channel_data)
        return resampled.astype(np.float32, copy=False)

    resampled_channels = [
        _resample(raw_signals[ch_idx], sample_rates[channel])
        for ch_idx, channel in enumerate(feature_columns)
    ]

    signals = np.vstack(resampled_channels)
    samples_per_epoch = int(target_rate * epoch_seconds)
    num_epochs_available = signals.shape[1] // samples_per_epoch
    if num_epochs_available == 0:
        raise ValueError(f"Insufficient samples in '{edf_path.name}' for even one epoch.")

    stage_labels = _parse_capslpdb_stage_file(annotation_path, epoch_seconds)
    if not stage_labels:
        raise ValueError(f"No valid sleep stage annotations found for '{annotation_path.name}'.")

    if len(stage_labels) < num_epochs_available:
        stage_labels.extend([stage_labels[-1]] * (num_epochs_available - len(stage_labels)))
    elif len(stage_labels) > num_epochs_available:
        stage_labels = stage_labels[:num_epochs_available]

    epochs = []
    labels = []
    for epoch_idx in range(num_epochs_available):
        start = epoch_idx * samples_per_epoch
        end = start + samples_per_epoch
        epoch_data = signals[:, start:end]
        label = stage_labels[epoch_idx]
        if label is None:
            continue
        epochs.append(epoch_data)
        labels.append(label)

    if not epochs:
        raise ValueError(f"Annotations for '{annotation_path.name}' did not align with usable epochs.")

    return np.stack(epochs).astype(np.float32), np.asarray(labels, dtype=np.int64)


class CapSleepDataset(Dataset):
    """Dataset that loads CAPSLPDB EDF pairs (.edf + .edf.st/.edf.atr)."""

    def __init__(
        self,
        edf_files: Sequence[str],
        sequence_length: int = 10,
        epoch_seconds: int = 30,
        sample_rate_hz: int = 100,
        feature_columns: Optional[Sequence[str]] = None,
    ):
        self.edf_files = list(edf_files)
        self.sequence_length = sequence_length
        self.epoch_seconds = epoch_seconds
        self.sample_rate_hz = sample_rate_hz
        self.samples_per_epoch = int(self.sample_rate_hz * self.epoch_seconds)
        self.feature_columns = list(feature_columns or DEFAULT_CAP_FEATURE_COLUMNS)

        self.sequences, self.labels = self._load_and_preprocess_data()
        if self.sequences.size == 0:
            raise ValueError(
                "No valid sequences were created from CAPSLPDB files. "
                "Verify that the EDF files have matching .st/.atr annotations "
                "and that the requested channels exist."
            )

        self.num_channels = self.sequences.shape[2]
        self.feature_dim = self.sequences.shape[2] * self.sequences.shape[3]

    def _find_annotation(self, edf_path: Path) -> Optional[Path]:
        for suffix in (".edf.st", ".edf.atr"):
            candidate = edf_path.with_suffix(suffix)
            if candidate.exists():
                return candidate
        return None

    def _load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        all_sequences: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        iterator: Iterable[str]
        print(f"CAPSLPDB files to process: {len(self.edf_files)}", flush=True)
        if len(self.edf_files) > 1:
            iterator = tqdm(self.edf_files, desc="preprocess-capslpdb", unit="file")
        else:
            iterator = self.edf_files

        for file_path_str in iterator:
            file_path = Path(file_path_str)
            annotation_path = self._find_annotation(file_path)
            if annotation_path is None:
                print(f"No annotation (.edf.st /.edf.atr) found for {file_path.name}. Skipping.", flush=True)
                continue

            print(f"\n--- Processing CAP record: {file_path.name} ---", flush=True)
            try:
                epochs, labels = _load_capslpdb_record(
                    file_path,
                    annotation_path,
                    self.feature_columns,
                    self.sample_rate_hz,
                    self.epoch_seconds,
                )
                print(
                    f"  epochs extracted: {epochs.shape[0]} | channels: {epochs.shape[1]} | samples/epoch: {epochs.shape[2]}",
                    flush=True,
                )
            except Exception as err:
                print(f"Failed to process {file_path.name}: {err}", flush=True)
                continue

            scaled_epochs = scale_epochs_channelwise(epochs)
            sequences, seq_labels = build_sequences(scaled_epochs, labels, self.sequence_length)
            if sequences.size == 0:
                continue
            all_sequences.append(sequences)
            all_labels.append(seq_labels)

        if not all_sequences:
            return (
                np.empty((0, self.sequence_length, len(self.feature_columns), self.samples_per_epoch), dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )

        concatenated_sequences = np.concatenate(all_sequences, axis=0)
        concatenated_labels = np.concatenate(all_labels, axis=0)
        return concatenated_sequences, concatenated_labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.sequences[idx]).to(torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
