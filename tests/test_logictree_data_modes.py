import pytest

from src.data.cifar10 import (
    CIFAR10_LOGICTREE_LARGE_BITPLANE_CHANNELS,
    CIFAR10_LOGICTREE_LARGE_CHANNELS,
    CIFAR10_LOGICTREE_SMALL_CHANNELS,
    get_cifar10_logictree_input_channels,
)
from src.data.loaders import validate_model_data_config
from src.models.model_factory import normalize_family


def test_logictree_small_channels():
    dcfg = {"dataset": "cifar10", "cifar10_input_mode": "small"}
    assert get_cifar10_logictree_input_channels(dcfg) == CIFAR10_LOGICTREE_SMALL_CHANNELS


def test_logictree_large_threshold31_channels():
    dcfg = {"dataset": "cifar10", "cifar10_input_mode": "large_threshold31"}
    assert get_cifar10_logictree_input_channels(dcfg) == CIFAR10_LOGICTREE_LARGE_CHANNELS


def test_logictree_large_bitplane5_channels():
    dcfg = {"dataset": "cifar10", "cifar10_input_mode": "large_bitplane5"}
    assert get_cifar10_logictree_input_channels(dcfg) == CIFAR10_LOGICTREE_LARGE_BITPLANE_CHANNELS


def test_validate_logictree_sets_expected_input_channels():
    mcfg = {
        "family": "logictreenet",
        "dataset": "cifar10",
        "num_classes": 10,
        "k": 512,
        "output_gate_factor": 2,
        "input_image_size": 32,
    }
    dcfg = {"dataset": "cifar10", "cifar10_input_mode": "small"}
    validate_model_data_config(mcfg, dcfg)
    assert mcfg["input_channels"] == CIFAR10_LOGICTREE_SMALL_CHANNELS


def test_normalize_family_only_logictreenet():
    assert normalize_family("logictreenet") == "logictreenet"
    assert normalize_family("LogicTreeNet") == "logictreenet"
    assert normalize_family(None) == "logictreenet"
    with pytest.raises(ValueError, match="Unknown model.family"):
        normalize_family("logictree")


def test_validate_rejects_unknown_family():
    mcfg = {
        "family": "logictree",
        "dataset": "cifar10",
        "num_classes": 10,
        "k": 32,
        "output_gate_factor": 1,
        "input_image_size": 32,
    }
    dcfg = {"dataset": "cifar10", "cifar10_input_mode": "small"}
    with pytest.raises(ValueError, match="Unknown model.family"):
        validate_model_data_config(mcfg, dcfg)


def test_validate_logictree_rejects_wrong_channels():
    mcfg = {
        "family": "logictreenet",
        "dataset": "cifar10",
        "num_classes": 10,
        "k": 512,
        "output_gate_factor": 2,
        "input_channels": 7,
        "input_image_size": 32,
    }
    dcfg = {"dataset": "cifar10", "cifar10_input_mode": "small"}
    with pytest.raises(ValueError, match="input_channels"):
        validate_model_data_config(mcfg, dcfg)
