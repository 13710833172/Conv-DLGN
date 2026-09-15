from src.config_profiles import apply_paper_table6_profile


def test_profile_cifar_g_values():
    cfg = {
        "model": {
            "family": "logictreenet",
            "dataset": "cifar10",
            "size": "G",
            "profile_source": "paper_table6",
            "num_classes": 10,
        },
        "data": {"dataset": "cifar10"},
        "training": {},
        "output": {},
    }
    out = apply_paper_table6_profile(cfg)
    assert out["model"]["k"] == 2560
    assert out["model"]["output_gate_factor"] == 1
    assert out["training"]["lr"] == 0.02
    assert out["training"]["weight_decay"] == 0.001
    assert out["training"]["batch_size"] == 128
    assert out["training"]["eval_every_steps"] == 2000
    assert out["training"]["log_every_steps"] == 100
    assert out["output"]["tau"] == 450.0
    assert out["data"]["cifar10_input_mode"] == "large_threshold31"


def test_profile_mnist_m_values():
    cfg = {
        "model": {
            "family": "logictreenet",
            "dataset": "mnist",
            "size": "M",
            "profile_source": "paper_table6",
            "num_classes": 10,
        },
        "data": {"dataset": "mnist"},
        "training": {},
        "output": {},
    }
    out = apply_paper_table6_profile(cfg)
    assert out["model"]["k"] == 64
    assert out["model"]["output_gate_factor"] == 2
    assert out["training"]["lr"] == 0.01
    assert out["training"]["weight_decay"] == 0.0
    assert out["training"]["batch_size"] == 256
    assert out["training"]["eval_every_steps"] == 500
    assert out["training"]["log_every_steps"] == 100
    assert out["output"]["tau"] == 28.0
