import argparse
import glob
import io
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from datasets.base import LABEL_MAP
from preprocess import DEFAULT_FEATURES, get_dataloader
from model import build_model

from typing import Mapping, Sequence


DEFAULT_INV_LABEL_MAP = {value: key for key, value in LABEL_MAP.items()}


class Tee(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        for stream in self.streams:
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def report_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str,
    ordered_labels: Sequence[int],
    inv_label_map: Mapping[int, str],
) -> None:
    """Print multi-class metrics for the given predictions."""
    if y_true is None or y_pred is None or y_true.size == 0:
        print(f"\n{split_name}: no samples available for metric reporting.")
        return

    accuracy = (y_true == y_pred).mean()
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    bal_accuracy = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"\n{split_name} metrics:")
    print(f"  Accuracy           : {accuracy:.4f}")
    print(f"  Balanced Accuracy  : {bal_accuracy:.4f}")
    print(f"  Macro F1           : {macro_f1:.4f}")
    print(f"  Weighted F1        : {weighted_f1:.4f}")
    print(f"  Cohen's Kappa      : {kappa:.4f}")
    print(f"  Matthews Corrcoef  : {mcc:.4f}")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=ordered_labels, zero_division=0
    )
    print("\n  Per-class metrics (precision | recall | f1 | support):")
    for label, p, r, f, s in zip(ordered_labels, precision, recall, f1, support):
        label_name = inv_label_map.get(label, str(label))
        print(f"    {label_name:>3} ({label}): {p:.4f} | {r:.4f} | {f:.4f} | {int(s)}")

    cm = confusion_matrix(y_true, y_pred, labels=ordered_labels)
    print("\n  Confusion matrix (rows=true, cols=pred):")
    print(cm)

    target_names = [inv_label_map.get(idx, str(idx)) for idx in ordered_labels]
    print("\n  Classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=ordered_labels,
            target_names=target_names,
            zero_division=0,
        )
    )


def summarize_normalized_features(dataset, feature_names):
    """Report per-channel stats after z-score normalization."""
    sequences = getattr(dataset, "sequences", None)
    if sequences is None or sequences.size == 0:
        print("Training dataset is empty; skipping normalized feature summary.")
        return
    axis = (0, 1, 3)
    means = sequences.mean(axis=axis)
    stds = sequences.std(axis=axis)
    mins = sequences.min(axis=axis)
    maxs = sequences.max(axis=axis)
    print("\nNormalized feature distribution (mean | std | min | max):")
    total_channels = means.shape[0]
    if not feature_names or len(feature_names) != total_channels:
        feature_names = [f"channel_{idx}" for idx in range(total_channels)]
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
        print(inputs.shape)
        exit()
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


def evaluate(model, dataloader, criterion, device, collect_stats: bool = False):
    running_loss = 0.0
    correct = 0
    total = 0
    collected_true = [] if collect_stats else None
    collected_pred = [] if collect_stats else None

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
            if collect_stats:
                collected_true.extend(labels.detach().cpu().numpy())
                collected_pred.extend(predicted.detach().cpu().numpy())
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total

    if collect_stats:
        return (
            epoch_loss,
            epoch_acc,
            np.asarray(collected_true, dtype=np.int64),
            np.asarray(collected_pred, dtype=np.int64),
        )
    return epoch_loss, epoch_acc, None, None


def run_pipeline(args):
    # Set GPU device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required but not available.")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"Using CUDA device {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    fix_seed(args.seed)

    dataset_name = args.data.lower()
    if dataset_name == "dreamt":
        default_dir = "/data/ceragem/physionet.org/files/dreamt/2.1.0/data_100Hz"
        data_dir = args.data_dir or default_dir
        pattern = os.path.join(data_dir, "*.csv")
        data_format = "dreamt"
        default_features = list(DEFAULT_FEATURES["dreamt"])

    all_files = sorted(glob.glob(pattern))
    if not all_files:
        raise FileNotFoundError(f"No data files found under {data_dir} (pattern: {pattern}).")

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
    if dataset_name == "sleepbrl" and args.sample_rate == 100:
        effective_sample_rate = 50
        print("Adjusting sample rate to 50 Hz for SleepBRL.")

    requested_features = [c.strip() for c in args.features.split(",") if c.strip()]
    if requested_features:
        feature_columns = requested_features
    elif default_features:
        feature_columns = default_features
    else:
        feature_columns = None

    if feature_columns:
        print(f"Using feature columns: {feature_columns}")
    else:
        print("Using dataset default feature columns.")

    train_loader = get_dataloader(train_files, args.batch_size, args.seq_len, shuffle=True, 
                                  epoch_seconds=args.epoch_seconds, sample_rate_hz=effective_sample_rate, 
                                  max_files=None, feature_columns=feature_columns,
                                  data_format=data_format)
    val_loader = get_dataloader(val_files, args.batch_size, args.seq_len, shuffle=False,
                                epoch_seconds=args.epoch_seconds, sample_rate_hz=effective_sample_rate,
                                max_files=None, feature_columns=feature_columns,
                                data_format=data_format)
    test_loader = get_dataloader(test_files, args.batch_size, args.seq_len, shuffle=False,
                                 epoch_seconds=args.epoch_seconds, sample_rate_hz=effective_sample_rate,
                                 max_files=None, feature_columns=feature_columns,
                                 data_format=data_format)

    train_dataset = train_loader.dataset
    inv_label_map = DEFAULT_INV_LABEL_MAP # Default inverse label map
    ordered_labels = sorted(inv_label_map.keys())
    
    if feature_columns:
        feature_names = list(feature_columns)
    else:
        feature_names = getattr(train_dataset, "feature_columns", None)
        
    summarize_normalized_features(train_dataset, feature_names)
    num_channels = getattr(train_dataset, "num_channels", train_dataset.sequences.shape[2])
    samples_per_epoch = getattr(train_dataset, "samples_per_epoch", effective_sample_rate * args.epoch_seconds)
    num_classes = getattr(train_dataset, "num_classes", None)
    if not num_classes:
        num_classes = max(ordered_labels) + 1 if ordered_labels else len(inv_label_map)
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

    losses, accs, val_macro_f1s = [], [], []
    best_state_dict = None
    best_epoch = -1
    best_val_macro_f1 = float("-inf")
    start_time = time.perf_counter()
    for epoch in tqdm(range(args.epochs), desc='ceragem'):
        train_loss, train_acc = train(model=model, 
                                      dataloader=train_loader,
                                      criterion=criterion, 
                                      optimizer=optimizer, 
                                      device=device)
        val_loss, val_acc, val_true, val_pred = evaluate(model=model, 
                                                         dataloader=val_loader, 
                                                         criterion=criterion, 
                                                         device=device,
                                                         collect_stats=True)
        if val_true is not None and val_true.size > 0:
            val_macro_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
        else:
            val_macro_f1 = float("nan")
        comparable_macro_f1 = val_macro_f1 if np.isfinite(val_macro_f1) else float("-inf")
        val_macro_f1s.append(val_macro_f1)
        if comparable_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = comparable_macro_f1
            best_epoch = epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        losses.append([train_loss, val_loss]); accs.append([train_acc, val_acc])
        # Pretty per-epoch logging for clarity
        val_f1_str = f"{val_macro_f1:.4f}" if np.isfinite(val_macro_f1) else "N/A"
        print(f"Epoch {epoch+1:>3}/{args.epochs}: "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | "
              f"val_macro_f1={val_f1_str}")

    # Summary table
    print("\nSummary (loss: train/val, acc: train/val):")
    for i, ((tr_l, v_l), (tr_a, v_a), v_f1) in enumerate(zip(losses, accs, val_macro_f1s), start=1):
        v_f1_str = f"{v_f1:.4f}" if np.isfinite(v_f1) else "N/A"
        print(f"  - Epoch {i:>3}: loss {tr_l:.4f}/{v_l:.4f} | acc {tr_a:.4f}/{v_a:.4f} | val_macro_f1 {v_f1_str}")
    elapsed = time.perf_counter() - start_time
    print(f"\nElapsed training time: {elapsed:.2f} seconds")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        best_f1_str = f"{best_val_macro_f1:.4f}" if np.isfinite(best_val_macro_f1) else "N/A"
        print(f"\nLoaded best model from epoch {best_epoch + 1} based on validation macro F1 {best_f1_str}.")

    val_loss_final, val_acc_final, val_true, val_pred = evaluate(model=model,
                                                                 dataloader=val_loader,
                                                                 criterion=criterion,
                                                                 device=device,
                                                                 collect_stats=True)
    if val_true is not None and val_true.size > 0:
        val_macro_f1_final = f1_score(val_true, val_pred, average="macro", zero_division=0)
    else:
        val_macro_f1_final = float("nan")
    val_macro_f1_final_str = f"{val_macro_f1_final:.4f}" if np.isfinite(val_macro_f1_final) else "N/A"
    print(f"\nFinal validation metrics: loss={val_loss_final:.4f} | acc={val_acc_final:.4f} | macro_f1={val_macro_f1_final_str}")
    report_classification_metrics(val_true, val_pred, "Validation", ordered_labels, inv_label_map)

    test_loss, test_acc, test_true, test_pred = evaluate(model=model,
                                                         dataloader=test_loader,
                                                         criterion=criterion,
                                                         device=device,
                                                         collect_stats=True)
    if test_true is not None and test_true.size > 0:
        test_macro_f1 = f1_score(test_true, test_pred, average="macro", zero_division=0)
    else:
        test_macro_f1 = float("nan")
    test_macro_f1_str = f"{test_macro_f1:.4f}" if np.isfinite(test_macro_f1) else "N/A"
    print(f"\nTest metrics: loss={test_loss:.4f} | acc={test_acc:.4f} | macro_f1={test_macro_f1_str}")
    report_classification_metrics(test_true, test_pred, "Test", ordered_labels, inv_label_map)


def main(args):
    log_file_handle = None
    tee = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        log_input = (args.log_file or "").strip()
        if log_input:
            log_path = Path(log_input).expanduser()
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_name = args.model
            model_slug = "".join(
                ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in model_name
            )
            if not model_slug.strip("_"):
                model_slug = "model"
            log_path = Path("logs") / f"{timestamp}_{model_slug}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file_handle = open(log_path, "w", encoding="utf-8")
        tee = Tee(original_stdout, log_file_handle)
        sys.stdout = tee
        sys.stderr = tee
        print(f"[Logging] Writing output to {log_path}")
        run_pipeline(args)
    finally:
        if tee:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        if log_file_handle:
            log_file_handle.close()


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
    data_group.add_argument('--max_files', type=int, default=None, help='Limit how many files to load per split (useful for quick experiments)')


    train_group = parser.add_argument_group("Training")
    train_group.add_argument('--batch_size', type=int, default=32)
    train_group.add_argument('--epochs', type=int, default=10)
    train_group.add_argument('--lr', type=float, default=0.01)
    train_group.add_argument('--seed', type=int, default=0)
    train_group.add_argument('--gpu', type=int, default=0, help='CUDA GPU index to use (0-based)')
    train_group.add_argument('--log_file', type=str, default='', help='File path to save console output (defaults to logs/<timestamp>_<model>.log)')

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
