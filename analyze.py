import argparse
import glob
import os
import random
import time
import warnings

from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from datasets.base import LABEL_MAP
from preprocess import get_dataloader
from model import build_model


EXCLUDED_COLUMNS = {
    "Sleep_Stage",
    "Obstructive_Apnea",
    "Central_Apnea",
    "Hypopnea",
    "Multiple_Events",
    "artifact",
    "sid",
    "TIMESTAMP",
    "timestamp",
    "Timestamp",
    "timestamp_start",
    "Timestamp_Start",
}

DEFAULT_INV_LABEL_MAP = {value: key for key, value in LABEL_MAP.items()}


def infer_dreamt_features(sample_csv: str) -> List[str]:
    """Infer all numeric DREAMT feature columns except labels/metadata."""
    try:
        df = pd.read_csv(sample_csv, nrows=1000)
    except pd.errors.EmptyDataError as err:
        raise ValueError(f"Failed to read sample file {sample_csv}: {err}") from err

    candidates = df.select_dtypes(include=["number"]).columns
    features = [col for col in candidates if col not in EXCLUDED_COLUMNS]
    if not features:
        raise ValueError(
            "No numeric feature columns found after excluding label/metadata columns. "
            "Check the input CSV structure."
        )
    return features


def summarize_training_split(train_dataset, feature_names: List[str]) -> None:
    """Print label distribution and per-feature stats for the training split."""
    labels = train_dataset.labels
    if labels.size == 0:
        print("Training dataset is empty; skipping summary.")
        return

    inv_label_map = getattr(train_dataset, "inv_label_map", DEFAULT_INV_LABEL_MAP)
    total = labels.size
    print("\nTraining dataset summary (60% split):")
    print(f"Number of features: {len(feature_names)}")
    raw_stage_counts = getattr(train_dataset, "stage_name_counts", None)
    if raw_stage_counts:
        total_raw = sum(raw_stage_counts.values())
        print("Raw stage distribution (count | percentage | stage):")
        preferred_order = ["W", "P", "N1", "N2", "N3", "R"]
        stages_to_print = []
        for stage in preferred_order:
            if stage in raw_stage_counts:
                stages_to_print.append(stage)
        for stage in sorted(raw_stage_counts.keys()):
            if stage not in stages_to_print:
                stages_to_print.append(stage)
        for stage in stages_to_print:
            count = raw_stage_counts[stage]
            pct = (count / total_raw) * 100 if total_raw else 0.0
            print(f"  {count:6d} | {pct:6.2f}% | {stage}")
    excluded_stage_counts = getattr(train_dataset, "excluded_stage_counts", None)
    if excluded_stage_counts:
        total_excluded = sum(excluded_stage_counts.values())
        if total_excluded > 0:
            print("Excluded stage distribution (not used for training/eval):")
            preferred_order = ["P", "W", "N1", "N2", "N3", "R"]
            stages_to_print = []
            for stage in preferred_order:
                if stage in excluded_stage_counts:
                    stages_to_print.append(stage)
            for stage in sorted(excluded_stage_counts.keys()):
                if stage not in stages_to_print:
                    stages_to_print.append(stage)
            for stage in stages_to_print:
                count = excluded_stage_counts[stage]
                pct = (count / total_excluded) * 100 if total_excluded else 0.0
                print(f"  {count:6d} | {pct:6.2f}% | {stage}")
    print("Label distribution (count | percentage | stage):")
    uniques, counts = np.unique(labels, return_counts=True)
    for label_id, count in zip(uniques, counts):
        pct = (count / total) * 100
        stage = inv_label_map.get(int(label_id), str(label_id))
        print(f"  {count:6d} | {pct:6.2f}% | {stage} ({label_id})")

    sequences = train_dataset.sequences
    axis = (0, 1, 3)
    means = sequences.mean(axis=axis)
    stds = sequences.std(axis=axis)
    mins = sequences.min(axis=axis)
    maxs = sequences.max(axis=axis)

    print("\nFeature distribution (mean | std | min | max):")
    for idx, name in enumerate(feature_names):
        print(
            f"  {name}: "
            f"{means[idx]: .4f} | "
            f"{stds[idx]: .4f} | "
            f"{mins[idx]: .4f} | "
            f"{maxs[idx]: .4f}"
        )


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def train(model, dataloader, criterion, optimizer, device):
    running_loss = 0.0
    correct = 0
    total = 0

    model.train()
    for batch in dataloader:
        if len(batch) == 3:
            inputs, lengths, labels = batch
        else:
            inputs, labels = batch
            lengths = None

        inputs = inputs.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device) if lengths is not None else None

        optimizer.zero_grad()
        outputs = model(inputs, lengths=lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    running_loss = 0.0
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                inputs, lengths, labels = batch
            else:
                inputs, labels = batch
                lengths = None

            inputs = inputs.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device) if lengths is not None else None

            outputs = model(inputs, lengths=lengths)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def main(args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    fix_seed(args.seed)

    if args.test_ratio != 0.1:
        warnings.warn("--test_ratio is deprecated and ignored; using 60/20/20 split.")

    dataset_name = args.data.lower()
    if dataset_name != "dreamt":
        raise ValueError("This script now supports only the DREAMT dataset.")

    default_dir = os.path.join(
        "/data/ceragem/physionet.org/files", "dreamt", "2.1.0", "data_100Hz"
    )
    data_dir = args.data_dir or default_dir
    pattern = os.path.join(data_dir, "*.csv")
    data_format = "dreamt"

    all_files = sorted(glob.glob(pattern))
    if not all_files:
        raise FileNotFoundError(f"No data files found under {data_dir} (pattern: {pattern}).")

    requested_features = [c.strip() for c in args.features.split(",") if c.strip()]
    if requested_features:
        feature_columns = requested_features
        print(f"Using user-provided feature columns ({len(feature_columns)} total).")
    else:
        inferred_features = infer_dreamt_features(all_files[0])
        feature_columns = inferred_features
        print(f"Inferred {len(feature_columns)} DREAMT feature columns from {all_files[0]}.")

    files = all_files[:]
    rnd = random.Random(args.seed)
    rnd.shuffle(files)
    if len(files) < 3:
        raise ValueError("At least 3 files are required to form 60/20/20 splits.")

    total_files = len(files)
    train_count = max(1, int(round(total_files * 0.6)))
    val_count = max(1, int(round(total_files * 0.2)))
    if train_count + val_count >= total_files:
        val_count = max(1, total_files - train_count - 1)
    test_count = total_files - train_count - val_count
    if test_count < 1:
        test_count = 1
        if train_count > val_count:
            train_count = max(1, train_count - 1)
        else:
            val_count = max(1, val_count - 1)

    assert train_count + val_count + test_count == total_files, "Split counts must sum to total files."

    train_files = files[:train_count]
    val_files = files[train_count : train_count + val_count]
    test_files = files[train_count + val_count :]

    print(
        f"Train files: {len(train_files)} | "
        f"Validation files: {len(val_files)} | "
        f"Test files: {len(test_files)}"
    )

    effective_sample_rate = args.sample_rate
    print(f"Using feature columns: {feature_columns}")

    train_loader = get_dataloader(train_files,
                                  args.batch_size,
                                  args.seq_len,
                                  shuffle=True,
                                  epoch_seconds=args.epoch_seconds,
                                  sample_rate_hz=effective_sample_rate,
                                #   max_files=args.max_files,
                                  max_files=None,
                                  
                                  feature_columns=feature_columns,
                                  data_format=data_format)
    val_loader = get_dataloader(val_files,
                                args.batch_size,
                                args.seq_len,
                                shuffle=False,
                                epoch_seconds=args.epoch_seconds,
                                sample_rate_hz=effective_sample_rate,
                                max_files=None,
                                feature_columns=feature_columns,
                                data_format=data_format)
    test_loader = get_dataloader(test_files,
                                 args.batch_size,
                                 args.seq_len,
                                 shuffle=False,
                                 epoch_seconds=args.epoch_seconds,
                                 sample_rate_hz=effective_sample_rate,
                                 max_files=None,
                                 feature_columns=feature_columns,
                                 data_format=data_format)

    train_dataset = train_loader.dataset
    summarize_training_split(train_dataset, feature_columns)

    inv_label_map = getattr(train_dataset, "inv_label_map", DEFAULT_INV_LABEL_MAP)
    ordered_label_ids = getattr(train_dataset, "ordered_label_ids", None)
    if not ordered_label_ids:
        ordered_label_ids = sorted(inv_label_map.keys())
    num_classes = getattr(train_dataset, "num_classes", None)
    if not num_classes:
        num_classes = max(ordered_label_ids) + 1 if ordered_label_ids else 1

    num_channels = getattr(train_dataset, "num_channels", train_dataset.sequences.shape[2])
    samples_per_epoch = getattr(train_dataset, "samples_per_epoch", effective_sample_rate * args.epoch_seconds)
    input_dim = getattr(train_dataset, "feature_dim", num_channels * samples_per_epoch)

    print(f"Using {num_channels} channels with {samples_per_epoch} samples per epoch.")

    if args.model.lower() == "lstm":
        model_kwargs = dict(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=num_classes,
            dropout=args.dropout,
        )
    elif args.model.lower() == "sleep_transformer":
        frame_stride = args.frame_stride if args.frame_stride > 0 else None
        model_kwargs = dict(
            num_channels=num_channels,
            samples_per_epoch=samples_per_epoch,
            seq_len=args.seq_len,
            num_classes=num_classes,
            frame_size=args.frame_size,
            frame_stride=frame_stride,
            frame_num_layers=args.frame_layers,
            frame_num_heads=args.frame_heads,
            frame_ff_dim=args.frame_ff_dim,
            seq_num_layers=args.seq_layers,
            seq_num_heads=args.seq_heads,
            seq_ff_dim=args.seq_ff_dim,
            attention_dim=args.attention_dim,
            fc_hidden_dim=args.fc_hidden_dim,
            dropout=args.transformer_dropout,
        )
    elif args.model.lower() == "deepsleepnet":
        model_kwargs = dict(
            num_channels=num_channels,
            samples_per_epoch=samples_per_epoch,
            n_classes=num_classes,
            seq_length=args.seq_len,
            n_rnn_layers=args.num_layers,      # LSTM의 num_layers 인자 재활용
            use_dropout=(args.dropout > 0.0)   # dropout > 0 이면 True로 설정
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    model = build_model(args.model, **model_kwargs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    losses, accs = [], []
    start_time = time.perf_counter()
    for epoch in tqdm(range(args.epochs), desc='ceragem'):
        train_loss, train_acc = train(model=model, 
                                      dataloader=train_loader,
                                      criterion=criterion, 
                                      optimizer=optimizer, 
                                      device=device)
        val_loss, val_acc = evaluate(model=model, 
                                     dataloader=val_loader, 
                                     criterion=criterion, 
                                     device=device)
        losses.append([train_loss, val_loss]); accs.append([train_acc, val_acc])
        # Pretty per-epoch logging for clarity
        print(f"Epoch {epoch+1:>3}/{args.epochs}: "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

    # Summary table
    print("\nSummary (loss: train/val, acc: train/val):")
    for i, ((tr_l, val_l), (tr_a, val_a)) in enumerate(zip(losses, accs), start=1):
        print(f"  - Epoch {i:>3}: loss {tr_l:.4f}/{val_l:.4f} | acc {tr_a:.4f}/{val_a:.4f}")
    elapsed = time.perf_counter() - start_time
    print(f"\nElapsed training time: {elapsed:.2f} seconds")

    test_loss, test_acc = evaluate(model=model,
                                   dataloader=test_loader,
                                   criterion=criterion,
                                   device=device)
    print(f"\nTest metrics: loss={test_loss:.4f} | acc={test_acc:.4f}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_num_threads(4)

    parser = argparse.ArgumentParser(description='ceragem')

    data_group = parser.add_argument_group("Data loading")
    data_group.add_argument('--data', type=str, default="dreamt")
    data_group.add_argument('--data_dir', type=str, default='', help='Override data directory path')
    data_group.add_argument('--features', type=str, default='', help='Comma-separated feature column list to override defaults')
    data_group.add_argument('--sample_rate', type=int, default=100, help='Samples per second in the raw data')
    data_group.add_argument('--epoch_seconds', type=int, default=30, help='Epoch length in seconds for windowing')
    data_group.add_argument('--seq_len', type=int, default=10)
    data_group.add_argument('--test_ratio', type=float, default=0.1, help='(Deprecated) Unused; splits are fixed to 60/20/20')
    data_group.add_argument('--max_files', type=int, default=None, help='Limit how many files to load per split (useful for quick experiments)')


    train_group = parser.add_argument_group("Training")
    train_group.add_argument('--batch_size', type=int, default=32)
    train_group.add_argument('--epochs', type=int, default=10)
    train_group.add_argument('--lr', type=float, default=0.01)
    train_group.add_argument('--seed', type=int, default=0)
    train_group.add_argument('--device', type=int, default=0)

    model_group = parser.add_argument_group("Model (LSTM/Transformer shared)")
    model_group.add_argument('--model', type=str, default='lstm', choices=['lstm', 'sleep_transformer', 'deepsleepnet'])
    model_group.add_argument('--dropout', type=float, default=0.5, help='Dropout for the LSTM model')
    model_group.add_argument('--num_layers', type=int, default=2)
    model_group.add_argument('--hidden_dim', type=int, default=32)

    transformer_group = parser.add_argument_group("Transformer-specific")
    transformer_group.add_argument('--frame_size', type=int, default=100, help='Frame size (samples) for SleepTransformer input windows')
    transformer_group.add_argument('--frame_stride', type=int, default=0, help='Frame stride (samples); 0 defaults to frame_size')
    transformer_group.add_argument('--frame_layers', type=int, default=2, help='Number of frame-level Transformer layers')
    transformer_group.add_argument('--frame_heads', type=int, default=8, help='Number of attention heads in frame-level Transformer')
    transformer_group.add_argument('--frame_ff_dim', type=int, default=1024, help='Feed-forward dimension in frame-level Transformer')
    transformer_group.add_argument('--seq_layers', type=int, default=2, help='Number of sequence-level Transformer layers')
    transformer_group.add_argument('--seq_heads', type=int, default=8, help='Number of attention heads in sequence-level Transformer')
    transformer_group.add_argument('--seq_ff_dim', type=int, default=1024, help='Feed-forward dimension in sequence-level Transformer')
    transformer_group.add_argument('--attention_dim', type=int, default=64, help='Additive attention hidden dimension')
    transformer_group.add_argument('--fc_hidden_dim', type=int, default=1024, help='Hidden dimension of the Transformer classifier head')
    transformer_group.add_argument('--transformer_dropout', type=float, default=0.1, help='Dropout used inside Transformer layers')

    args = parser.parse_args()

    main(args)
