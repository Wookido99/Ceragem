"""Dataset factory utilities."""

from __future__ import annotations

from typing import Sequence

from .dreamt import DEFAULT_DREAMT_FEATURES, DreamtDataset
from .capslpdb import DEFAULT_CAP_FEATURE_COLUMNS, CapSleepDataset
from .sleepbrl import DEFAULT_SLEEPBRL_FEATURES, SleepBrlDataset

DATASET_REGISTRY = {
    "dreamt": (DreamtDataset, DEFAULT_DREAMT_FEATURES),
    "capslpdb": (CapSleepDataset, DEFAULT_CAP_FEATURE_COLUMNS),
    "sleepbrl": (SleepBrlDataset, DEFAULT_SLEEPBRL_FEATURES),
}


def resolve_dataset(name: str):
    key = name.lower()
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[key]


__all__ = [
    "DreamtDataset",
    "CapSleepDataset",
    "SleepBrlDataset",
    "DEFAULT_DREAMT_FEATURES",
    "DEFAULT_CAP_FEATURE_COLUMNS",
    "DEFAULT_SLEEPBRL_FEATURES",
    "resolve_dataset",
]
