"""Unified data loader entry for LogicTreeNet-only setup."""

from __future__ import annotations

from src.data.cifar10 import get_cifar10_loaders, get_cifar10_logictree_input_channels
from src.data.mnist import get_mnist_loaders
from src.models.model_factory import normalize_family


def _norm_dataset(name: str) -> str:
    return str(name).lower().replace("-", "").replace("_", "")


_CIFAR10_INPUT_MODES = frozenset({"small", "large_threshold31", "large_bitplane5"})


def cifar10_input_mode(dcfg: dict) -> str:
    """Normalize and validate ``data.cifar10_input_mode``."""
    raw = dcfg.get("cifar10_input_mode", "small")
    s = str(raw).lower().replace("-", "_")
    if s not in _CIFAR10_INPUT_MODES:
        raise ValueError(
            "data.cifar10_input_mode must be 'small', 'large_threshold31', "
            f"or 'large_bitplane5', got {raw!r}"
        )
    return s


def _ensure_logictree_input_shape(mcfg: dict, dcfg: dict) -> None:
    ds = _norm_dataset(dcfg.get("dataset", mcfg.get("dataset", "mnist")))
    if ds == "mnist":
        mcfg.setdefault("input_channels", 1)
        mcfg.setdefault("input_image_size", 28)
        if int(mcfg["input_channels"]) != 1:
            raise ValueError("MNIST LogicTreeNet expects input_channels == 1")
        if int(mcfg["input_image_size"]) != 28:
            raise ValueError("MNIST LogicTreeNet expects input_image_size == 28")
        return

    if ds == "cifar10":
        mode = cifar10_input_mode(dcfg)
        expected_ch = get_cifar10_logictree_input_channels(dcfg)
        mcfg.setdefault("input_channels", expected_ch)
        mcfg.setdefault("input_image_size", 32)
        if int(mcfg["input_channels"]) != expected_ch:
            raise ValueError(
                f"CIFAR-10 mode={mode!r} expects input_channels={expected_ch}, "
                f"got {mcfg['input_channels']}."
            )
        if int(mcfg["input_image_size"]) != 32:
            raise ValueError("CIFAR-10 LogicTreeNet expects input_image_size == 32")
        return

    raise ValueError(
        f"Unknown data.dataset {dcfg.get('dataset')!r}; expected 'mnist' or 'cifar10'."
    )


def validate_model_data_config(mcfg: dict, dcfg: dict) -> None:
    normalize_family(mcfg.get("family", "logictreenet"))
    ds = _norm_dataset(dcfg.get("dataset", mcfg.get("dataset", "mnist")))

    if int(mcfg.get("k", 0)) <= 0:
        raise ValueError("LogicTreeNet expects model.k > 0")
    if int(mcfg.get("output_gate_factor", 0)) <= 0:
        raise ValueError("LogicTreeNet expects model.output_gate_factor > 0")
    if int(mcfg["k"]) % int(mcfg["output_gate_factor"]) != 0:
        raise ValueError("LogicTreeNet expects model.k divisible by model.output_gate_factor")
    _ensure_logictree_input_shape(mcfg, dcfg)
    mcfg["dataset"] = ds


def get_data_loaders(
    dcfg: dict,
    batch_size: int,
    num_workers: int | None = None,
    model_family: str = "logictreenet",
    for_teacher: bool = False,
) -> tuple:
    """Return (train_loader, val_loader, test_loader)."""
    normalize_family(model_family)
    ds = _norm_dataset(dcfg.get("dataset", "mnist"))
    nw = int(dcfg.get("num_workers", 2)) if num_workers is None else int(num_workers)
    data_dir = dcfg.get("data_dir", "data")
    val_split = int(dcfg.get("val_split", 5000))
    data_seed = int(dcfg.get("data_seed", 42))

    if ds == "mnist":
        return get_mnist_loaders(
            batch_size=batch_size,
            data_dir=data_dir,
            binarize_threshold=float(dcfg.get("binarize_threshold", 0.5)),
            val_split=val_split,
            data_seed=data_seed,
            num_workers=nw,
        )

    if ds == "cifar10":
        mode = cifar10_input_mode(dcfg)
        return get_cifar10_loaders(
            batch_size=batch_size,
            data_dir=data_dir,
            val_split=val_split,
            data_seed=data_seed,
            num_workers=nw,
            input_mode=mode,
            for_teacher=for_teacher,
        )

    raise ValueError(f"Unknown data.dataset {dcfg.get('dataset')!r}")
