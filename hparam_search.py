#!/usr/bin/env python
"""Lightweight hyperparameter search driver for `main.py`.

This script sweeps over specified hyperparameters by launching `main.py`
sub-processes, captures their console output, and aggregates validation/test
metrics (including macro F1) into a summary JSON file.

Example usage:
    python hparam_search.py --data dreamt --data-dir /path/to/dreamt \
        --models lstm sleep_transformer \
        --seq-lens 8 12 16 --epoch-seconds 20 30 \
        --lrs 0.0005 0.0008 --dropouts 0.3 0.5 --hidden-dims 64 128 \
        --batch-sizes 32 --epochs 10 --gpu 0
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


VAL_PATTERN = re.compile(
    r"Final validation metrics:\s+loss=([0-9.]+)\s+\|\s+acc=([0-9.]+)\s+\|\s+macro_f1=([0-9NAnan\.\-]+)"
)
TEST_PATTERN = re.compile(
    r"Test metrics:\s+loss=([0-9.]+)\s+\|\s+acc=([0-9.]+)\s+\|\s+macro_f1=([0-9NAnan\.\-]+)"
)
BEST_EPOCH_PATTERN = re.compile(
    r"Loaded best model from epoch\s+(\d+)\s+based on validation macro F1\s+([0-9NAnan\.\-]+)"
)


def _safe_float(token: str) -> Optional[float]:
    token = token.strip()
    if token.lower() in {"n/a", "na", "nan"}:
        return None
    try:
        return float(token)
    except ValueError:
        return None


@dataclass
class RunResult:
    run_index: int
    command: List[str]
    log_file: Path
    returncode: int
    hyperparams: Dict[str, float]
    val_loss: Optional[float] = None
    val_acc: Optional[float] = None
    val_macro_f1: Optional[float] = None
    test_loss: Optional[float] = None
    test_acc: Optional[float] = None
    test_macro_f1: Optional[float] = None
    best_epoch: Optional[int] = None
    best_epoch_macro_f1: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "run_index": self.run_index,
            "command": " ".join(self.command),
            "log_file": str(self.log_file),
            "returncode": self.returncode,
            "hyperparams": self.hyperparams,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "val_macro_f1": self.val_macro_f1,
            "test_loss": self.test_loss,
            "test_acc": self.test_acc,
            "test_macro_f1": self.test_macro_f1,
            "best_epoch": self.best_epoch,
            "best_epoch_macro_f1": self.best_epoch_macro_f1,
        }
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter search orchestrator for main.py")
    parser.add_argument("--data", default="dreamt")
    parser.add_argument("--data-dir", default="", help="Optional dataset directory override")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--extra-args",
        default="",
        help="Additional CLI arguments to forward to main.py (as a single string)",
    )

    parser.add_argument("--models", nargs="+", default=["lstm", "sleep_transformer", "deepsleepnet"])
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[10, 20, 30], dest="seq_lens")
    parser.add_argument(
        "--epoch-seconds",
        nargs="+",
        type=int,
        default=[20, 30, 60],
        dest="epoch_seconds_list",
    )
    parser.add_argument("--lrs", nargs="+", type=float, default=[1e-3], dest="learning_rates")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32], dest="batch_sizes")
    parser.add_argument("--dropouts", nargs="+", type=float, default=[0.3])
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[64, 128], dest="hidden_dims")

    parser.add_argument("--max-runs", type=int, default=0, help="Stop after N runs (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing")
    parser.add_argument("--log-dir", default="logs/hparam_search")
    parser.add_argument("--summary-dir", default="results/hparam_search")
    return parser.parse_args()


def build_grid(args: argparse.Namespace) -> Iterable[Dict[str, object]]:
    keys = ["model", "seq_len", "epoch_seconds", "lr", "batch_size", "dropout", "hidden_dim"]
    space = itertools.product(
        args.models,
        args.seq_lens,
        args.epoch_seconds_list,
        args.learning_rates,
        args.batch_sizes,
        args.dropouts,
        args.hidden_dims,
    )
    for combo in space:
        yield dict(zip(keys, combo))


def build_command(
    base_python: str,
    base_cli: List[str],
    hyperparams: Dict[str, object],
    log_file: Path,
) -> List[str]:
    cmd = [base_python, "main.py"] + base_cli
    cmd.extend(["--model", str(hyperparams["model"])])
    cmd.extend(["--seq_len", str(hyperparams["seq_len"])])
    cmd.extend(["--epoch_seconds", str(hyperparams["epoch_seconds"])])
    cmd.extend(["--lr", str(hyperparams["lr"])])
    cmd.extend(["--batch_size", str(hyperparams["batch_size"])])
    cmd.extend(["--dropout", str(hyperparams["dropout"])])
    cmd.extend(["--hidden_dim", str(hyperparams["hidden_dim"])])

    # Keep Transformer dropout in sync when applicable.
    cmd.extend(["--transformer_dropout", str(hyperparams["dropout"])])

    cmd.extend(["--log_file", str(log_file)])
    return cmd


def parse_metrics(output: str, run_result: RunResult) -> None:
    if match := VAL_PATTERN.search(output):
        run_result.val_loss = _safe_float(match.group(1))
        run_result.val_acc = _safe_float(match.group(2))
        run_result.val_macro_f1 = _safe_float(match.group(3))
    if match := TEST_PATTERN.search(output):
        run_result.test_loss = _safe_float(match.group(1))
        run_result.test_acc = _safe_float(match.group(2))
        run_result.test_macro_f1 = _safe_float(match.group(3))
    if match := BEST_EPOCH_PATTERN.search(output):
        run_result.best_epoch = int(match.group(1))
        run_result.best_epoch_macro_f1 = _safe_float(match.group(2))


def main() -> None:
    args = parse_args()
    base_cli: List[str] = [
        "--data",
        args.data,
        "--epochs",
        str(args.epochs),
        "--seed",
        str(args.seed),
        "--gpu",
        str(args.gpu),
    ]
    if args.data_dir:
        base_cli.extend(["--data_dir", args.data_dir])
    if args.extra_args:
        base_cli.extend(shlex.split(args.extra_args))

    log_dir = Path(args.log_dir)
    summary_dir = Path(args.summary_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    results: List[RunResult] = []
    timestamp_global = datetime.now().strftime("%Y%m%d-%H%M%S")

    grid_iter = enumerate(build_grid(args), start=1)
    for run_index, hyperparams in grid_iter:
        if args.max_runs and run_index > args.max_runs:
            break
        timestamp_run = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_name = (
            f"{timestamp_run}_run{run_index:03d}_{hyperparams['model']}_"
            f"sl{hyperparams['seq_len']}_es{hyperparams['epoch_seconds']}_"
            f"lr{hyperparams['lr']}_bs{hyperparams['batch_size']}.log"
        )
        log_file = log_dir / log_name
        cmd = build_command(sys.executable, base_cli, hyperparams, log_file)

        print(f"[Run {run_index}] {' '.join(cmd)}")
        if args.dry_run:
            continue

        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        if completed.stdout:
            # main.py already writes to log_file, but ensure we capture stdout/stderr in case
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write("\n\n# --- Captured stdout ---\n")
                lf.write(completed.stdout)
                if completed.stderr:
                    lf.write("\n\n# --- Captured stderr ---\n")
                    lf.write(completed.stderr)

        run_result = RunResult(
            run_index=run_index,
            command=cmd,
            log_file=log_file,
            returncode=completed.returncode,
            hyperparams=hyperparams,
        )
        parse_metrics(completed.stdout or "", run_result)
        results.append(run_result)

        status = "OK" if completed.returncode == 0 else f"RC={completed.returncode}"
        print(
            f"    -> status: {status}, "
            f"val_macro_f1={run_result.val_macro_f1}, "
            f"test_macro_f1={run_result.test_macro_f1}"
        )

    if args.dry_run:
        print("Dry run complete. No commands were executed.")
        return

    summary_payload = {
        "generated_at": timestamp_global,
        "total_runs": len(results),
        "results": [r.to_dict() for r in results],
    }
    summary_path = summary_dir / f"hparam_summary_{timestamp_global}.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2)
    print(f"\nWrote summary to {summary_path}")

    successful = [r for r in results if r.returncode == 0 and r.val_macro_f1 is not None]
    if successful:
        top = sorted(successful, key=lambda r: r.val_macro_f1 or -1, reverse=True)[:5]
        print("\nTop runs by validation macro F1:")
        for item in top:
            print(
                f"  Run {item.run_index:03d}: val_macro_f1={item.val_macro_f1:.4f} | "
                f"test_macro_f1={(item.test_macro_f1 or float('nan')):.4f} | "
                f"params={item.hyperparams} | log={item.log_file}"
            )
    else:
        print("\nNo successful runs with parsed validation macro F1 found.")


if __name__ == "__main__":
    main()
