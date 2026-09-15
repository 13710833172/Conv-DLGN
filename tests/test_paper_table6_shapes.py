import torch
import yaml
from pathlib import Path

from src.config_profiles import apply_paper_table6_profile
from src.data.loaders import validate_model_data_config
from src.models.model_factory import build_soft_model_from_config
from src.utils.output_grouping import compute_class_scores


EXPECTED = {
    "cifar10_s.yaml": {
        "input": (9, 32, 32),
        "pool_shapes": [(32, 16, 16), (128, 8, 8), (512, 4, 4), (1024, 2, 2)],
        "flatten": 4096,
        "heads": [40960, 20480, 10240],
    },
    "cifar10_m.yaml": {
        "input": (9, 32, 32),
        "pool_shapes": [(256, 16, 16), (1024, 8, 8), (4096, 4, 4), (8192, 2, 2)],
        "flatten": 32768,
        "heads": [327680, 163840, 81920],
    },
    "cifar10_b.yaml": {
        "input": (15, 32, 32),
        "pool_shapes": [(512, 16, 16), (2048, 8, 8), (8192, 4, 4), (16384, 2, 2)],
        "flatten": 65536,
        "heads": [1310720, 655360, 327680],
    },
    "cifar10_l.yaml": {
        "input": (93, 32, 32),
        "pool_shapes": [(1024, 16, 16), (4096, 8, 8), (16384, 4, 4), (32768, 2, 2)],
        "flatten": 131072,
        "heads": [2621440, 1310720, 655360],
    },
    "cifar10_g.yaml": {
        "input": (93, 32, 32),
        "pool_shapes": [(2560, 16, 16), (10240, 8, 8), (40960, 4, 4), (81920, 2, 2)],
        "flatten": 327680,
        "heads": [3276800, 1638400, 819200],
    },
    "mnist_s.yaml": {
        "input": (1, 28, 28),
        "pool_shapes": [(16, 12, 12), (48, 6, 6), (144, 3, 3)],
        "flatten": 1296,
        "heads": [40960, 20480, 10240],
    },
    "mnist_m.yaml": {
        "input": (1, 28, 28),
        "pool_shapes": [(64, 12, 12), (192, 6, 6), (576, 3, 3)],
        "flatten": 5184,
        "heads": [163840, 81920, 40960],
    },
    "mnist_l.yaml": {
        "input": (1, 28, 28),
        "pool_shapes": [(256, 12, 12), (768, 6, 6), (2304, 3, 3)],
        "flatten": 20736,
        "heads": [327680, 163840, 81920],
    },
}


def _load_model_from_config(config_path: Path):
    cfg = yaml.safe_load(config_path.read_text())
    cfg = apply_paper_table6_profile(cfg)
    mcfg, dcfg = cfg["model"], cfg["data"]
    validate_model_data_config(mcfg, dcfg)
    model = build_soft_model_from_config(mcfg, connectivity_seed=int(cfg["checkpoint"].get("seed", 0)))
    return model, mcfg


def test_paper_table6_backbone_and_head_shapes():
    cfg_dir = Path("configs/paper_table6")
    for filename, exp in EXPECTED.items():
        model, mcfg = _load_model_from_config(cfg_dir / filename)
        bsz = 1
        c, h, w = exp["input"]
        x = torch.rand(bsz, c, h, w)

        with torch.no_grad():
            z = x
            for i, (conv, pool) in enumerate(zip(model.conv_layers, model.pool_layers, strict=True)):
                z = pool(conv(z))
                assert tuple(z.shape[1:]) == exp["pool_shapes"][i], (filename, i, tuple(z.shape), exp["pool_shapes"][i])

        flat = z.view(z.size(0), -1)
        assert flat.shape == (bsz, exp["flatten"]), (filename, tuple(flat.shape), exp["flatten"])
        assert model.backbone_feature_dim == exp["flatten"], (filename, model.backbone_feature_dim, exp["flatten"])

        assert len(model.head_layers) == 3
        assert model.head_layers[0].input_dim == exp["flatten"]
        assert model.head_layers[0].num_neurons == exp["heads"][0]
        assert model.head_layers[1].input_dim == exp["heads"][0]
        assert model.head_layers[1].num_neurons == exp["heads"][1]
        assert model.head_layers[2].input_dim == exp["heads"][1]
        assert model.head_layers[2].num_neurons == exp["heads"][2]
        assert model.head_width == exp["heads"][2]

        fake_head_out = torch.rand(bsz, exp["heads"][2])
        logits = compute_class_scores(fake_head_out, num_classes=int(mcfg["num_classes"]), tau=1.0, beta=0.0)
        assert logits.shape == (bsz, 10), (filename, tuple(logits.shape))
