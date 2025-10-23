"""DREAMT CSV based dataset loader."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .base import LABEL_MAP, build_sequences, load_csv_with_sleep_labels, scale_epochs_channelwise

DEFAULT_DREAMT_FEATURES: Sequence[str] = (
    "SNORE",
    "PTAF",
    "FLOW"
)


class DreamtDataset(Dataset):
    """Loads DREAMT CSV files and returns windowed sleep stage data."""

    def __init__(
        self,
        csv_files: Sequence[str],
        sequence_length: int = 10,
        epoch_seconds: int = 30,
        sample_rate_hz: int = 100,
        feature_columns: Optional[Sequence[str]] = None,
    ):
        self.csv_files = list(csv_files)
        self.sequence_length = sequence_length
        self.label_column = "Sleep_Stage"
        self.sample_rate_hz = sample_rate_hz
        self.epoch_seconds = epoch_seconds
        self.samples_per_epoch = int(self.sample_rate_hz * self.epoch_seconds)
        self.feature_columns = list(feature_columns or DEFAULT_DREAMT_FEATURES)

        self.sequences, self.labels = self._load_and_preprocess_data()
        if self.sequences.size == 0:
            raise ValueError(
                "No valid sequences were created. Verify the feature list, epoch length, "
                "and that the source CSVs contain sufficient data."
            )

        self.num_channels = self.sequences.shape[2]
        self.feature_dim = self.sequences.shape[2] * self.sequences.shape[3]

    def _load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        all_epochs: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        iterator: Iterable[str]
        if len(self.csv_files) > 1:
            iterator = tqdm(self.csv_files, desc="preprocess-dreamt", unit="file")
        else:
            iterator = self.csv_files

        for file_path_str in iterator:
            file_path = Path(file_path_str)
            print(f"\n--- Processing file: {file_path} ---")
            try:
                epochs, labels = load_csv_with_sleep_labels(
                    file_path,
                    self.feature_columns,
                    self.label_column,
                    self.samples_per_epoch,
                )
            except ValueError as err:
                print(f"Skipping file due to preprocessing error: {err}")
                continue
            if epochs.size == 0:
                continue
            scaled = scale_epochs_channelwise(epochs)
            sequences, seq_labels = build_sequences(scaled, labels, self.sequence_length)
            if sequences.size == 0:
                continue
            all_epochs.append(sequences)
            all_labels.append(seq_labels)

        if not all_epochs:
            return (
                np.empty((0, self.sequence_length, len(self.feature_columns), self.samples_per_epoch), dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )

        return np.concatenate(all_epochs, axis=0), np.concatenate(all_labels, axis=0)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.sequences[idx]).to(torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
