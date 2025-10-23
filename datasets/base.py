"""Shared utilities for dataset preprocessing."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

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
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a DREAMT CSV file and return epoch arrays and labels."""
    try:
        df = pd.read_csv(file_path)
    except ParserError:
        df = pd.read_csv(file_path, engine="python", on_bad_lines="skip")

    df = df.replace({label_column: {"P": "W"}})
    df = df[df[label_column] != "Missing"]
    df[label_column] = df[label_column].map(LABEL_MAP)
    df = df.dropna(subset=[label_column] + list(feature_columns))
    df = df.reset_index(drop=True)
    df["epoch_id"] = (df.index // samples_per_epoch).astype(int)

    epochs: List[np.ndarray] = []
    labels: List[int] = []
    for _, group in df.groupby("epoch_id"):
        if len(group) != samples_per_epoch:
            continue
        epochs.append(group[feature_columns].to_numpy(dtype=np.float32, copy=True).T)
        labels.append(int(group[label_column].iloc[0]))

    return np.stack(epochs) if epochs else np.empty((0, len(feature_columns), samples_per_epoch), dtype=np.float32), np.asarray(labels, dtype=np.int64)
