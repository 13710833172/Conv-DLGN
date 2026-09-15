"""Evaluate soft/hard models from a checkpoint on the test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_profiles import apply_profile_if_requested
from src.data.loaders import get_data_loaders, validate_model_data_config
from src.models.model_factory import (
    build_hard_model_from_soft,
    build_soft_model_from_config,
    normalize_family,
)
from src.utils.metrics import discretization_gap
from src.utils.output_grouping import compute_class_scores

@torch.no_grad()
def acc_model(forward_fn, loader, device, num_classes, tau, beta):
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = forward_fn(x)
        logits = compute_class_scores(out, num_classes, tau, beta)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    ckpt = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    cfg = apply_profile_if_requested(ckpt["config"])
    mcfg, dcfg, ocfg = cfg["model"], cfg["data"], cfg["output"]
    ccfg = cfg.get("checkpoint", {})
    conn_seed = int(ckpt.get("connectivity_seed", ccfg.get("seed", 0)))
    family = normalize_family(mcfg.get("family", "logictreenet"))

    validate_model_data_config(mcfg, dcfg)
    _, _, test_loader = get_data_loaders(
        dcfg=dcfg,
        batch_size=int(cfg["training"]["batch_size"]),
        model_family=family,
    )

    soft = build_soft_model_from_config(mcfg, connectivity_seed=conn_seed)
    soft.load_state_dict(ckpt["model_state_dict"], strict=True)
    soft.to(device)
    soft.eval()

    hard = build_hard_model_from_soft(soft, mcfg).to(device)
    hard.eval()

    nc = int(mcfg["num_classes"])
    tau = float(ocfg["tau"])
    beta = float(ocfg.get("beta", 0.0))

    s_acc = acc_model(soft, test_loader, device, nc, tau, beta)
    h_acc = acc_model(hard, test_loader, device, nc, tau, beta)
    gap = discretization_gap(s_acc, h_acc)
    print(f"family: {family}")
    print(f"test_soft_acc: {s_acc:.6f}")
    print(f"test_hard_acc: {h_acc:.6f}")
    print(f"discretization_gap (test_soft_acc - test_hard_acc): {gap:.6f}")


if __name__ == "__main__":
    main()
