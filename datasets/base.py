"""Shared utilities for dataset preprocessing."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.errors import ParserError

LABEL_MAP = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}


def canonical_channel(label: str) -> str:
    """Normalize channel labels for matching (case/spacing insensitive)."""
    return re.sub(r"[^a-z0-9]+", "", label.lower())


def scale_epochs_channelwise(epochs: np.ndarray) -> np.ndarray:
    """Apply per-channel standard scaling to (num_epochs, C, samples) array."""
    scaled = epochs.astype(np.float32, copy=True)
    for ch in range(scaled.shape[1]):
        channel = scaled[:, ch, :]
        mean = float(channel.mean())
        std = float(channel.std())
        if std < 1e-6:
            std = 1.0
        scaled[:, ch, :] = (channel - mean) / std
    return scaled


def build_sequences(
    epochs: np.ndarray, labels: np.ndarray, sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct sliding sequences with associated labels."""
    sequences: List[np.ndarray] = []
    sequence_labels: List[int] = []
    windowable = max(0, len(epochs) - sequence_length + 1)
    for start in range(windowable):
        sequences.append(epochs[start : start + sequence_length])
        sequence_labels.append(labels[start + sequence_length - 1])
    if not sequences:
        shape = (0, sequence_length, epochs.shape[1], epochs.shape[2])
        return np.empty(shape, dtype=np.float32), np.empty(0, dtype=np.int64)
    stacked = np.stack(sequences).astype(np.float32)
    return stacked, np.asarray(sequence_labels, dtype=np.int64)


def load_csv_with_sleep_labels(
    file_path: Path,
    feature_columns: Sequence[str],
    label_column: str,
    samples_per_epoch: int,
    label_map: Optional[Mapping[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load a DREAMT CSV file and return epoch arrays, numeric labels, kept stages, and all stages."""
    try:
        df = pd.read_csv(file_path)
    except ParserError:
        df = pd.read_csv(file_path, engine="python", on_bad_lines="skip")

    label_lookup = label_map or LABEL_MAP
    df = df[df[label_column] != "Missing"]
    df["stage_original"] = df[label_column].astype(str)
    df[label_column] = df[label_column].map(label_lookup)
    df = df.dropna(subset=[label_column] + list(feature_columns))
    df = df.reset_index(drop=True)
    df["epoch_id"] = (df.index // samples_per_epoch).astype(int)

    epochs: List[np.ndarray] = []
    labels: List[int] = []
    stage_names_kept: List[str] = []
    stage_names_all: List[str] = []
    for _, group in df.groupby("epoch_id"):
        if len(group) != samples_per_epoch:
            continue
        stage_name = str(group["stage_original"].iloc[0])
        stage_names_all.append(stage_name)
        label_value = group[label_column].iloc[0]
        if pd.isna(label_value):
            continue
        epochs.append(group[feature_columns].to_numpy(dtype=np.float32, copy=True).T)
        labels.append(int(label_value))
        stage_names_kept.append(stage_name)

    if epochs:
        epoch_array = np.stack(epochs)
    else:
        epoch_array = np.empty((0, len(feature_columns), samples_per_epoch), dtype=np.float32)

    return epoch_array, np.asarray(labels, dtype=np.int64), stage_names_kept, stage_names_all
