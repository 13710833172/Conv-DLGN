"""Paper Table-6 profiles for Convolutional DLGN."""

from __future__ import annotations


CIFAR10_TABLE6 = {
    "S": {"k": 32, "tau": 20.0, "lr": 0.02, "weight_decay": 0.002, "batch_size": 128, "ox": 1},
    "M": {"k": 256, "tau": 40.0, "lr": 0.02, "weight_decay": 0.002, "batch_size": 128, "ox": 1},
    "B": {"k": 512, "tau": 280.0, "lr": 0.02, "weight_decay": 0.002, "batch_size": 16, "ox": 2},
    "L": {"k": 1024, "tau": 340.0, "lr": 0.02, "weight_decay": 0.002, "batch_size": 128, "ox": 2},
    "G": {"k": 2560, "tau": 450.0, "lr": 0.02, "weight_decay": 0.001, "batch_size": 128, "ox": 1},
}

MNIST_TABLE6 = {
    "S": {"k": 16, "tau": 6.5, "lr": 0.01, "weight_decay": 0.0, "batch_size": 512, "ox": 2},
    "M": {"k": 64, "tau": 28.0, "lr": 0.01, "weight_decay": 0.0, "batch_size": 256, "ox": 2},
    "L": {"k": 256, "tau": 35.0, "lr": 0.01, "weight_decay": 0.0, "batch_size": 128, "ox": 1},
}


def _norm_dataset(name: str) -> str:
    return str(name).lower().replace("-", "").replace("_", "")


def _norm_size(size: str) -> str:
    return str(size).strip().upper()


def apply_paper_table6_profile(cfg: dict) -> dict:
    """In-place default filling using Table-6 profile values."""
    mcfg = cfg.setdefault("model", {})
    dcfg = cfg.setdefault("data", {})
    tcfg = cfg.setdefault("training", {})
    ocfg = cfg.setdefault("output", {})

    ds = _norm_dataset(mcfg.get("dataset", dcfg.get("dataset", "")))
    if ds not in ("mnist", "cifar10"):
        raise ValueError(f"model.dataset/data.dataset must be mnist or cifar10, got {ds!r}")
    size = _norm_size(mcfg.get("size", ""))
    if not size:
        raise ValueError("model.size is required for profile_source=paper_table6")

    table = CIFAR10_TABLE6 if ds == "cifar10" else MNIST_TABLE6
    if size not in table:
        raise ValueError(f"Unsupported size {size!r} for dataset {ds}.")
    p = table[size]

    mcfg.setdefault("family", "logictreenet")
    mcfg["dataset"] = ds
    mcfg["size"] = size
    mcfg["profile_source"] = "paper_table6"
    mcfg.setdefault("num_classes", 10)
    mcfg.setdefault("k", int(p["k"]))
    mcfg.setdefault("output_gate_factor", int(p["ox"]))
    mcfg.setdefault("receptive_field_size", 3)
    mcfg.setdefault("gate_init", "residual_a")
    mcfg.setdefault("num_head_layers", 3)
    mcfg.setdefault("tree_stride", 1)
    mcfg.setdefault("output_neuron_factor", 320)

    tcfg.setdefault("optimizer", "adamw")
    tcfg.setdefault("lr", float(p["lr"]))
    tcfg.setdefault("weight_decay", float(p["weight_decay"]))
    tcfg.setdefault("batch_size", int(p["batch_size"]))
    tcfg.setdefault("max_epochs", 200)
    tcfg.setdefault("device", "cuda")
    # MNIST trains ~100–400 steps/epoch; 5000 was too sparse (~50 epochs between evals on S).
    tcfg.setdefault("eval_every_steps", 2000 if ds == "cifar10" else 500)
    tcfg.setdefault("log_every_steps", 100)

    ocfg.setdefault("tau", float(p["tau"]))
    ocfg.setdefault("beta", 0.0)

    dcfg["dataset"] = ds
    dcfg.setdefault("data_seed", 42)
    dcfg.setdefault("data_dir", "data")
    dcfg.setdefault("num_workers", 2)
    if ds == "cifar10":
        dcfg.setdefault("val_split", 5000)
        if size in ("S", "M"):
            dcfg.setdefault("cifar10_input_mode", "small")
        else:
            dcfg.setdefault("cifar10_input_mode", "large_threshold31")
        mcfg.setdefault("input_image_size", 32)
    else:
        dcfg.setdefault("val_split", 10000)
        dcfg.setdefault("binarize_threshold", 0.5)
        mcfg.setdefault("input_image_size", 28)
        mcfg.setdefault("input_channels", 1)

    return cfg
