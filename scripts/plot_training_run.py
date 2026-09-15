#!/usr/bin/env python3
"""Plot training curves with **epoch** on the x-axis (CSV rows only, no densification).

Outputs (default next to ``train_log.csv``, or under ``--out`` directory; if ``--out``
ends with ``.png``, the parent directory is used with fixed basenames):

- ``training_loss.png`` from ``train_epoch_log.csv`` (one mean train loss per epoch).
- ``val_acc.png`` from ``train_log.csv`` (one point per validation row, x = epoch).
- ``layer_stats.png`` from ``layer_stats.csv`` when present (x = epoch).

Examples:
  python scripts/plot_training_run.py --run-dir logs/mnist_s/run_20260514_051421
  python scripts/plot_training_run.py --train-log path/to/train_log.csv \\
      --epoch-log path/to/train_epoch_log.csv --out path/to/plot_dir/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

VAL_ACC_COLUMNS = ("epoch", "val_soft_acc", "val_hard_acc")

LAYER_STATS_READ_COLUMNS = (
    "epoch",
    "layer_idx",
    "mean_gate_entropy",
    "mean_max_gate_probability",
)


def _read_csv_columns(path: Path, columns: tuple[str, ...]) -> dict[str, list[float]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty or invalid CSV header: {path}")
        missing = [c for c in columns if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}; expected {list(columns)}")
        out: dict[str, list[float]] = {c: [] for c in columns}
        for lineno, row in enumerate(reader, start=2):
            for col in columns:
                try:
                    out[col].append(float(row[col]))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}: line {lineno}: bad value for {col!r}: {row.get(col)!r}") from exc
    if not out[columns[0]]:
        raise ValueError(f"No data rows in {path}")
    return out


def _epoch_loss_column(fieldnames: list[str] | None) -> str:
    if not fieldnames:
        raise ValueError("epoch log CSV has no header row")
    names = set(fieldnames)
    for key in ("train_loss", "epoch_train_loss", "soft_train_loss"):
        if key in names:
            return key
    raise ValueError(
        "train_epoch_log.csv must contain one of: 'train_loss', 'epoch_train_loss', 'soft_train_loss'. "
        f"Found columns: {fieldnames!r}."
    )


def _read_epoch_train_log(path: Path) -> tuple[list[float], list[float]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "epoch" not in reader.fieldnames:
            raise ValueError(f"{path}: missing required column 'epoch'")
        loss_col = _epoch_loss_column(list(reader.fieldnames))
        epochs: list[float] = []
        losses: list[float] = []
        for lineno, row in enumerate(reader, start=2):
            try:
                epochs.append(float(row["epoch"]))
                losses.append(float(row[loss_col]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}: line {lineno}: invalid row {row!r}") from exc
    if not epochs:
        raise ValueError(f"No data rows in {path}")
    if any(epochs[i] <= epochs[i - 1] for i in range(1, len(epochs))):
        raise ValueError(f"{path}: epoch must be strictly increasing")
    return epochs, losses


def _resolve_out_dir(train_log_path: Path, out_arg: Path | None) -> Path:
    if out_arg is None:
        return train_log_path.parent
    out_arg = out_arg.expanduser().resolve()
    if out_arg.suffix.lower() == ".png":
        return out_arg.parent
    return out_arg


def _plot_epoch_training_loss(epochs: list[float], losses: list[float], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, losses, color="C0", marker="o", ms=4, linestyle="-", linewidth=0.9)
    ax.set_xlabel("epoch")
    ax.set_ylabel("epoch train loss (mean over batches)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_val_acc_epoch(data: dict[str, list[float]], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ep = data["epoch"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ep, data["val_soft_acc"], marker="o", ms=4, linestyle="-", linewidth=0.8, label="val_soft_acc")
    ax.plot(ep, data["val_hard_acc"], marker="s", ms=4, linestyle="-", linewidth=0.8, label="val_hard_acc")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _read_layer_stats_by_layer(path: Path) -> dict[int, dict[str, list[float]]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    by_layer: dict[int, dict[str, list[float]]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty or invalid CSV header: {path}")
        missing = [c for c in LAYER_STATS_READ_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")

        for lineno, row in enumerate(reader, start=2):
            try:
                li = int(float(row["layer_idx"]))
                ep = float(row["epoch"])
                ent = float(row["mean_gate_entropy"])
                mp = float(row["mean_max_gate_probability"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}: line {lineno}: invalid row {row!r}") from exc
            if li not in by_layer:
                by_layer[li] = {"epoch": [], "entropy": [], "maxp": []}
            by_layer[li]["epoch"].append(ep)
            by_layer[li]["entropy"].append(ent)
            by_layer[li]["maxp"].append(mp)

    if not by_layer:
        raise ValueError(f"No data rows in {path}")
    return by_layer


def _plot_layer_stats_epoch(by_layer: dict[int, dict[str, list[float]]], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title)

    for li in sorted(by_layer):
        d = by_layer[li]
        pairs = sorted(zip(d["epoch"], d["entropy"], d["maxp"]))
        ex = [p[0] for p in pairs]
        ey = [p[1] for p in pairs]
        my = [p[2] for p in pairs]
        ax0.plot(ex, ey, marker="o", ms=2, linestyle="-", linewidth=0.8, label=f"layer {li}")
        ax1.plot(ex, my, marker="o", ms=2, linestyle="-", linewidth=0.8, label=f"layer {li}")

    ax0.set_ylabel("mean_gate_entropy")
    ax0.legend(loc="upper right", fontsize=7, ncol=2)
    ax0.grid(True, alpha=0.3)

    ax1.set_ylabel("mean_max_gate_probability")
    ax1.set_xlabel("epoch")
    ax1.legend(loc="lower right", fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot training logs (epoch x-axis, CSV points only)")
    ap.add_argument("--run-dir", type=Path, default=None, help="Run directory (train_log.csv inside)")
    ap.add_argument("--train-log", type=Path, default=None, help="Path to train_log.csv")
    ap.add_argument(
        "--epoch-log",
        type=Path,
        default=None,
        help="Path to train_epoch_log.csv (default: next to train_log or under --run-dir)",
    )
    ap.add_argument(
        "--layer-stats",
        type=Path,
        default=None,
        help="Path to layer_stats.csv (default: next to train_log or under --run-dir)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory, or any path ending in .png (parent dir used; fixed PNG names)",
    )
    ap.add_argument(
        "--no-layer-stats",
        action="store_true",
        help="Do not read/plot layer_stats.csv even if present",
    )
    args = ap.parse_args()

    if args.run_dir and args.train_log:
        print("error: pass only one of --run-dir or --train-log", file=sys.stderr)
        sys.exit(2)
    if not args.run_dir and not args.train_log:
        print("error: require --run-dir or --train-log", file=sys.stderr)
        sys.exit(2)

    if args.run_dir:
        run_dir = args.run_dir.resolve()
        train_log_path = run_dir / "train_log.csv"
        default_epoch = run_dir / "train_epoch_log.csv"
        default_layer = run_dir / "layer_stats.csv"
        title_prefix = str(run_dir)
    else:
        train_log_path = args.train_log.resolve()
        default_epoch = train_log_path.with_name("train_epoch_log.csv")
        default_layer = train_log_path.parent / "layer_stats.csv"
        title_prefix = str(train_log_path.parent)

    epoch_log_path = args.epoch_log.resolve() if args.epoch_log else default_epoch
    layer_path = args.layer_stats.resolve() if args.layer_stats else default_layer

    out_dir = _resolve_out_dir(train_log_path, args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    loss_out = out_dir / "training_loss.png"
    val_out = out_dir / "val_acc.png"
    layer_out = out_dir / "layer_stats.png"

    epoch_log_ok = False
    if epoch_log_path.is_file():
        try:
            ex, ly = _read_epoch_train_log(epoch_log_path)
            _plot_epoch_training_loss(
                ex,
                ly,
                loss_out,
                title=f"Train loss (epoch mean)\n{title_prefix}",
            )
            print("Wrote", loss_out)
            epoch_log_ok = True
        except (OSError, ValueError) as exc:
            print(f"error: cannot plot training loss from {epoch_log_path}: {exc}", file=sys.stderr)
    else:
        print(
            "error: train_epoch_log.csv not found at "
            f"{epoch_log_path}. Cannot write training_loss.png.\n"
            "  Re-run training (writes one row per epoch) or pass --epoch-log to a valid CSV.\n"
            "  Columns required: epoch, and one of train_loss / epoch_train_loss / soft_train_loss.",
            file=sys.stderr,
        )

    try:
        val_data = _read_csv_columns(train_log_path, VAL_ACC_COLUMNS)
    except (OSError, ValueError) as exc:
        print(f"error: cannot read validation columns from train_log: {exc}", file=sys.stderr)
        sys.exit(1)

    _plot_val_acc_epoch(val_data, val_out, title=f"Validation accuracy\n{title_prefix}")
    print("Wrote", val_out)

    if not args.no_layer_stats and layer_path.is_file():
        try:
            by_layer = _read_layer_stats_by_layer(layer_path)
            _plot_layer_stats_epoch(by_layer, layer_out, title=f"Layer stats\n{title_prefix}")
            print("Wrote", layer_out)
        except (OSError, ValueError) as exc:
            print(f"warning: skip layer_stats: {exc}", file=sys.stderr)
    elif not args.no_layer_stats:
        print("No layer_stats.csv found; skipped layer_stats.png")

    if not epoch_log_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
