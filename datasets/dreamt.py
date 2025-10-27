"""DREAMT CSV based dataset loader."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .base import LABEL_MAP, build_sequences, load_csv_with_sleep_labels, scale_epochs_channelwise

DEFAULT_DREAMT_FEATURES: Sequence[str] = (
    "C4-M1", "F4-M1", "O2-M1", "Fp1-O2", "T3 - CZ",
    "CZ - T4", 
    # "CHIN", 
    "E1", "E2", "ECG",
    # "LAT", "RAT", "SNORE", 
    "PTAF", "FLOW",
    # "THORAX", 
    "ABDOMEN", "SAO2", "BVP", "ACC_X",
    "ACC_Y", "ACC_Z", "TEMP", "EDA", "HR",
    "IBI"
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
        self.label_map = {
            "W": 0,
            "N1": 1,
            "N2": 2,
            "N3": 3,
            "R": 4,
        }
        self.inv_label_map = {value: key for key, value in self.label_map.items()}
        self.ordered_label_ids = sorted(self.inv_label_map.keys())
        self.num_classes = len(self.ordered_label_ids)

        self.stage_name_counts: Counter[str] = Counter()
        self.excluded_stage_counts: Counter[str] = Counter()
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
        stage_counter_sequences: Counter[str] = Counter()
        stage_counter_all: Counter[str] = Counter()
        stage_counter_excluded: Counter[str] = Counter()

        iterator: Iterable[str]
        if len(self.csv_files) > 1:
            iterator = tqdm(self.csv_files, desc="preprocess-dreamt", unit="file")
        else:
            iterator = self.csv_files

        for file_path_str in iterator:
            file_path = Path(file_path_str)
            print(f"\n--- Processing file: {file_path} ---")
            try:
                epochs, labels, stage_names_kept, stage_names_all = load_csv_with_sleep_labels(
                    file_path,
                    self.feature_columns,
                    self.label_column,
                    self.samples_per_epoch,
                    self.label_map,
                )
            except ValueError as err:
                print(f"Skipping file due to preprocessing error: {err}")
                continue
            if epochs.size == 0:
                continue
            scaled = scale_epochs_channelwise(epochs)
            sequences, seq_labels = build_sequences(scaled, labels, self.sequence_length)
            stage_counter_all.update(stage_names_all)
            excluded = [name for name in stage_names_all if name not in stage_names_kept]
            stage_counter_excluded.update(excluded)
            if sequences.size > 0 and stage_names_kept:
                windowable = len(stage_names_kept) - self.sequence_length + 1
                if windowable > 0:
                    for start in range(windowable):
                        stage_counter_sequences.update(
                            [stage_names_kept[start + self.sequence_length - 1]]
                        )
            if sequences.size == 0:
                continue
            all_epochs.append(sequences)
            all_labels.append(seq_labels)

        if not all_epochs:
            return (
                np.empty((0, self.sequence_length, len(self.feature_columns), self.samples_per_epoch), dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )

        self.stage_name_counts = stage_counter_all
        self.excluded_stage_counts = stage_counter_excluded
        return np.concatenate(all_epochs, axis=0), np.concatenate(all_labels, axis=0)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.sequences[idx]).to(torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )
