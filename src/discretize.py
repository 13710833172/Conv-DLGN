"""Export discretized (argmax) logic bundles from training checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_profiles import apply_profile_if_requested
from src.data.loaders import validate_model_data_config
from src.models.model_factory import build_soft_model_from_config, normalize_family
from src.models.logic_tree_net import LogicTreeNet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = apply_profile_if_requested(ckpt["config"])
    mcfg = cfg["model"]
    dcfg = cfg["data"]
    ocfg = cfg["output"]
    ccfg = cfg.get("checkpoint", {})
    family = normalize_family(mcfg.get("family", "logictreenet"))

    conn_seed = int(ckpt.get("connectivity_seed", ccfg.get("seed", 0)))
    validate_model_data_config(mcfg, dcfg)
    model = build_soft_model_from_config(mcfg, connectivity_seed=conn_seed)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    out_path = Path(args.out) if args.out else ckpt_path.with_name(ckpt_path.stem + "_discrete.pt")

    if not isinstance(model, LogicTreeNet):
        raise TypeError("Expected LogicTreeNet for family=logictreenet")
    payload = {
        "family": family,
        "connectivity_seed": conn_seed,
        "conv_leaf_indices": [x.cpu() for x in model.conv_leaf_indices()],
        "conv_gate_ids": [x.cpu() for x in model.conv_gate_ids()],
        "head_connectivities": [x.cpu() for x in model.head_connectivity_tensors()],
        "head_gate_ids": [x.cpu() for x in model.head_gate_ids()],
        "meta": {
            "dataset": mcfg.get("dataset"),
            "size": mcfg.get("size"),
            "k": int(mcfg["k"]),
            "output_gate_factor": int(mcfg["output_gate_factor"]),
            "num_classes": int(mcfg["num_classes"]),
            "input_channels": int(mcfg["input_channels"]),
            "input_image_size": int(mcfg["input_image_size"]),
            "num_conv_blocks": int(model.num_conv_blocks),
            "num_head_layers": int(model.num_head_layers),
            "head_width": int(model.head_width),
            "receptive_field_size": int(model.receptive_field_size),
            "tau": float(ocfg["tau"]),
            "beta": float(ocfg.get("beta", 0.0)),
            "source_checkpoint": str(ckpt_path),
        },
    }

    torch.save(payload, out_path)
    print("Saved discretized bundle to", out_path)


if __name__ == "__main__":
    main()
