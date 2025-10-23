"""DataLoader factory bridging dataset-specific implementations."""

from __future__ import annotations

from typing import Optional, Sequence

from torch.utils.data import DataLoader

try:
    from DREAMT_FE.datasets import (
        DEFAULT_DREAMT_FEATURES,
        # DEFAULT_CAP_FEATURE_COLUMNS,
        # DEFAULT_SLEEPBRL_FEATURES,
        # CapSleepDataset,
        # DreamtDataset,
        # SleepBrlDataset,
        resolve_dataset,
    )
except ImportError:  # fallback when executed as script from package directory
    from datasets import (
        DEFAULT_DREAMT_FEATURES,
        # DEFAULT_CAP_FEATURE_COLUMNS,
        # DEFAULT_SLEEPBRL_FEATURES,
        # CapSleepDataset,
        # DreamtDataset,
        # SleepBrlDataset,
        resolve_dataset,
    )

DEFAULT_FEATURES = {
    "dreamt": DEFAULT_DREAMT_FEATURES,
    # "capslpdb": DEFAULT_CAP_FEATURE_COLUMNS,
    # "sleepbrl": DEFAULT_SLEEPBRL_FEATURES,
}

__all__ = [
    "get_dataloader",
    "DEFAULT_FEATURES",
]


def get_dataloader(
    files: Sequence[str],
    batch_size: int = 32,
    sequence_length: int = 10,
    shuffle: bool = True,
    epoch_seconds: int = 30,
    sample_rate_hz: int = 100,
    max_files: Optional[int] = None,
    feature_columns: Optional[Sequence[str]] = None,
    data_format: str = "dreamt",
) -> DataLoader:
    """Construct a PyTorch DataLoader for the requested dataset format."""
    file_list = list(files)
    if max_files is not None:
        file_list = file_list[:max_files]

    dataset_cls, defaults = resolve_dataset(data_format)
    if feature_columns:
        features = list(feature_columns)
    elif defaults:
        features = list(defaults)
    else:
        features = None

    dataset_kwargs = dict(
        sequence_length=sequence_length,
        epoch_seconds=epoch_seconds,
        sample_rate_hz=sample_rate_hz,
        feature_columns=features,
    )
    dataset = dataset_cls(file_list, **dataset_kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
