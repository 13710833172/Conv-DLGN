"""Model construction helpers for LogicTreeNet-only workflow."""

from __future__ import annotations

from typing import Any

from src.models.hard_logic_tree_net import HardLogicTreeNet
from src.models.logic_tree_net import LogicTreeNet


def normalize_family(name: str | None) -> str:
    """Normalize and validate ``model.family`` for the logictreenet-only workflow."""
    if name is None:
        return "logictreenet"
    s = str(name).strip().lower().replace("-", "_")
    if s != "logictreenet":
        raise ValueError(
            f"Unknown model.family {name!r}; expected exactly 'logictreenet'."
        )
    return "logictreenet"


def build_soft_model_from_config(mcfg: dict[str, Any], connectivity_seed: int):
    normalize_family(mcfg.get("family", "logictreenet"))
    return LogicTreeNet(
        dataset=str(mcfg["dataset"]),
        input_channels=int(mcfg["input_channels"]),
        input_image_size=int(mcfg["input_image_size"]),
        k=int(mcfg["k"]),
        output_gate_factor=int(mcfg["output_gate_factor"]),
        num_classes=int(mcfg["num_classes"]),
        connectivity_seed=int(connectivity_seed),
        receptive_field_size=int(mcfg.get("receptive_field_size", 3)),
        gate_init=str(mcfg.get("gate_init", "residual_a")),
        num_head_layers=int(mcfg.get("num_head_layers", 3)),
        tree_stride=int(mcfg.get("tree_stride", 1)),
        output_neuron_factor=int(mcfg.get("output_neuron_factor", 320)),
    )


def build_hard_model_from_soft(soft_model, mcfg: dict[str, Any]):
    normalize_family(mcfg.get("family", "logictreenet"))
    return HardLogicTreeNet.from_soft_model(soft_model)
