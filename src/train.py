"""Train LogicTreeNet with soft/hard evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import nullcontext
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

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
from src.utils.metrics import (
    collect_soft_layer_stats,
    discretization_gap,
    soft_hard_layerwise_agreement,
)
from src.utils.output_grouping import compute_class_scores


def load_config(path: Path) -> dict:
    """加载YAML配置文件"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML config must be a mapping")
    return cfg

def resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_str == "cuda":
        print("CUDA not available, using CPU")
    return torch.device("cpu")


def build_optimizer(model: nn.Module, tcfg: dict) -> torch.optim.Optimizer:
    opt_name = str(tcfg.get("optimizer", "adam")).lower()
    lr = float(tcfg.get("lr", 0.01))
    wd = float(tcfg.get("weight_decay", 0.0))
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer {opt_name!r}; expected adam or adamw.")


def load_teacher_model_if_enabled(teacher_cfg: dict, num_classes: int, device: torch.device):
    if not bool(teacher_cfg.get("enabled", False)):
        return None

    ckpt_raw = teacher_cfg.get("checkpoint", "")
    if not ckpt_raw:
        raise ValueError("teacher.enabled=true requires teacher.checkpoint")
    ckpt_path = Path(ckpt_raw)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"teacher checkpoint not found: {ckpt_path}")

    arch = str(teacher_cfg.get("arch", "")).strip()
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if arch:
        from torchvision import models  # 如果配置了arch（网络架构名），则尝试用torchvision.models创建对应模型，并加载权重

        try:
            arch_name = arch.lower()
            if arch_name in ("resnet18", "resnet18_cifar", "cifar_resnet18"):
                teacher = models.resnet18(weights=None, num_classes=num_classes)
                teacher.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # 将第一层卷积核由默认的 7x7 改为 3x3，步幅为 1，padding 为 1
                teacher.maxpool = nn.Identity() # 去掉最大池化层
            elif hasattr(models, arch): # 如果模型架构在 torchvision.models 中
                teacher = getattr(models, arch)(num_classes=num_classes) # 获取模型架构
            else:
                teacher = models.get_model(arch, weights=None, num_classes=num_classes) # 获取模型架构
        except Exception as exc:  # pragma: no cover - defensive path
            raise ValueError(f"Failed to construct teacher architecture {arch!r}") from exc

        state = obj.get("model_state_dict", obj.get("state_dict", obj)) if isinstance(obj, dict) else obj
        if not isinstance(state, dict):
            raise ValueError("teacher checkpoint is not a state_dict mapping")
        teacher.load_state_dict(
            {
                (k[7:] if isinstance(k, str) and k.startswith("module.") else k): v
                for k, v in state.items()
            },
            strict=False,
        )
    elif isinstance(obj, nn.Module):
        teacher = obj
    elif isinstance(obj, dict) and isinstance(obj.get("model"), nn.Module):
        teacher = obj["model"]
    else:
        raise ValueError(
            "teacher.checkpoint must contain a full nn.Module when teacher.arch is empty"
        )

    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


@torch.no_grad()
def evaluate_soft(
    model: nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    tau: float,
    beta: float,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        logits = compute_class_scores(out, num_classes, tau, beta)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_hard(
    hard: nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    tau: float,
    beta: float,
) -> float:
    hard.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = hard(x)
        logits = compute_class_scores(out, num_classes, tau, beta)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_soft_hard_pair(
    model: nn.Module,
    hard: nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    tau: float,
    beta: float,
) -> tuple[float, float, float]:
    soft_acc = evaluate_soft(model, loader, device, num_classes, tau, beta)
    hard_acc = evaluate_hard(hard, loader, device, num_classes, tau, beta)
    return soft_acc, hard_acc, discretization_gap(soft_acc, hard_acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = apply_profile_if_requested(load_config(cfg_path))

    mcfg = cfg["model"]
    dcfg = cfg["data"]
    tcfg = cfg["training"]
    ocfg = cfg["output"]
    lcfg = cfg.get("logging", {})
    ccfg = cfg.get("checkpoint", {})
    teacher_cfg = cfg.get("teacher", {})
    # Per-run student checkpoints are saved under run_dir/checkpoints/; ccfg["save_dir"] is not used.

    family = normalize_family(mcfg.get("family", "logictreenet"))
    device = resolve_device(str(tcfg.get("device", "cuda")))
    conn_seed = int(ccfg.get("seed", 0))

    validate_model_data_config(mcfg, dcfg)
    train_loader, val_loader, test_loader = get_data_loaders(
        dcfg=dcfg,
        batch_size=int(tcfg["batch_size"]),
        model_family=family,
        for_teacher=bool(teacher_cfg.get("enabled", False)),
    )

    model = build_soft_model_from_config(mcfg, connectivity_seed=conn_seed).to(device)
    optimizer = build_optimizer(model, tcfg)
    criterion = nn.CrossEntropyLoss()
    teacher = load_teacher_model_if_enabled(teacher_cfg, int(mcfg["num_classes"]), device)
    # Distillation temperature for KL(student, teacher); independent from output.tau.
    kd_temperature = float(
        teacher_cfg.get(
            "kd_temperature",
            teacher_cfg.get("temperature_scale", teacher_cfg.get("temperature", math.sqrt(2.0))),
        )
    )
    teacher_weight = float(teacher_cfg.get("distill_weight", 0.5))
    if not (0.0 <= teacher_weight <= 1.0):
        raise ValueError("teacher.distill_weight must be in [0,1]")

    max_epochs = int(tcfg["max_epochs"])
    num_classes = int(mcfg["num_classes"])
    tau = float(ocfg["tau"])
    beta = float(ocfg.get("beta", 0.0))
    eval_every_steps = int(tcfg.get("eval_every_steps", 0))
    if eval_every_steps <= 0:
        eval_every_steps = (
            2000  # CIFAR: validate every 2000 steps
            if str(mcfg.get("dataset", dcfg.get("dataset", "mnist"))).lower().startswith("cifar")
            else 500  # MNIST: ~1–5 epochs between evals (see paper_table6 profile)
        )
    log_every_steps = int(tcfg.get("log_every_steps", 100))
    if log_every_steps <= 0:
        raise ValueError("training.log_every_steps must be >= 1")

    evaluate_test_during_training = bool(lcfg.get("evaluate_test_during_training", False)) # 是否在训练过程中评估测试集
    log_soft_hard = bool(lcfg.get("log_soft_hard_consistency", False)) # 是否记录软硬一致性指标
    cons_interval = int(lcfg.get("consistency_interval", 1)) # 每多少个epoch记录一次软硬一致性指标
    log_step_loss = bool(lcfg.get("log_step_loss", True))  # 是否写入 train_step_log.csv（稀疏）
    step_loss_log_flush_every = int(lcfg.get("step_loss_log_flush_every", 100))
    if step_loss_log_flush_every < 0:
        raise ValueError("training.step_loss_log_flush_every must be >= 0 (0 = flush every step)")
    step_loss_log_interval = int(lcfg.get("step_loss_log_interval", 100))
    if step_loss_log_interval < 1:
        raise ValueError("training.step_loss_log_interval must be >= 1")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ds_name = str(mcfg.get("dataset", dcfg.get("dataset", "unknown"))).strip().lower().replace("-", "_")
    size_name = str(mcfg.get("size", "custom")).strip().lower()
    run_group = f"{ds_name}_{size_name}"
    run_dir = Path("logs") / run_group / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_ckpt_dir = run_dir / "checkpoints"
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_ckpt_dir / "best_model.pt"
    last_path = run_ckpt_dir / "last_model.pt"

    train_log_path = run_dir / "train_log.csv"
    train_epoch_log_path = run_dir / "train_epoch_log.csv"
    train_step_log_path = run_dir / "train_step_log.csv"
    layer_stats_path = run_dir / "layer_stats.csv"
    final_test_path = run_dir / "final_test.json"
    summary_path = run_dir / "run_summary.json"

    with open(train_log_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(
            [
                "epoch",
                "global_step",
                "train_loss_since_eval",
                "train_acc_since_eval",
                "val_soft_acc",
                "val_hard_acc",
                "val_gap",
                "lr",
                "elapsed_seconds",
            ]
        )
    with open(layer_stats_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(
            [
                "epoch",
                "global_step",
                "layer_idx",
                "mean_gate_entropy",
                "mean_max_gate_probability",
                "activation_mean",
                "activation_std",
            ]
        )

    with open(train_epoch_log_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["epoch", "train_loss"])

    best_val_soft = -1.0
    best_val_hard = -1.0
    best_val_gap = 0.0
    best_epoch = -1
    best_step = -1
    global_step = 0
    train_start = time.time()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    def run_validation(epoch: int, step: int):
        """验证模型在验证集上的表现"""
        nonlocal best_val_soft, best_val_hard, best_val_gap, best_epoch, best_step # 更新最佳验证指标
        nonlocal running_loss, running_correct, running_total # 更新运行指标

        lr = float(optimizer.param_groups[0]["lr"]) 
        elapsed = time.time() - train_start # 计算运行时间
        train_loss = running_loss / max(running_total, 1) # 计算训练损失
        train_acc = running_correct / max(running_total, 1)

        hard = build_hard_model_from_soft(model, mcfg).to(device)
        val_soft_acc, val_hard_acc, val_gap = evaluate_soft_hard_pair(
            model, hard, val_loader, device, num_classes, tau, beta
        )

        # 计算并记录最新的训练损失、准确率、验证软/硬准确率和gap等，并写入train_log.csv
        with open(train_log_path, "a", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(
                [
                    epoch,
                    step,
                    f"{train_loss:.6f}",
                    f"{train_acc:.6f}",
                    f"{val_soft_acc:.6f}",
                    f"{val_hard_acc:.6f}",
                    f"{val_gap:.6f}",
                    f"{lr:.8f}",
                    f"{elapsed:.3f}",
                ]
            )

        # 调用collect_soft_layer_stats收集软门层的统计特征，写入layer_stats.csv
        model.eval()
        xb, _ = next(iter(val_loader))
        xb = xb.to(device, non_blocking=True)
        gate_stats, act_stats = collect_soft_layer_stats(model, xb)
        with open(layer_stats_path, "a", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            for li, ((h, mp), (am, ast)) in enumerate(zip(gate_stats, act_stats)):
                writer.writerow(
                    [
                        epoch,
                        step,
                        li,
                        f"{h:.6f}",
                        f"{mp:.6f}",
                        f"{am:.6f}",
                        f"{ast:.6f}",
                    ]
                )

        msg = (
            f"epoch={epoch}/{max_epochs} step={step} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_soft={val_soft_acc:.4f} val_hard={val_hard_acc:.4f} val_gap={val_gap:.4f}"
        )

        if evaluate_test_during_training:
            test_soft_acc, test_hard_acc, test_gap = evaluate_soft_hard_pair(
                model, hard, test_loader, device, num_classes, tau, beta
            )
            msg += (
                f" test_soft={test_soft_acc:.4f} test_hard={test_hard_acc:.4f} "
                f"test_gap={test_gap:.4f}"
            )
        print(msg)

        if log_soft_hard and (epoch % max(cons_interval, 1) == 0):
            x_small = xb[: min(64, xb.size(0))]
            rates = soft_hard_layerwise_agreement(model, hard, x_small)
            print("  soft_hard_layer_agree:", [round(r, 4) for r in rates])

        # 如果当前验证软准确率为最佳，则保存模型与相关状态为checkpoint。
        if val_soft_acc > best_val_soft:
            best_val_soft = float(val_soft_acc)
            best_val_hard = float(val_hard_acc)
            best_val_gap = float(val_gap)
            best_epoch = int(epoch)
            best_step = int(step)
            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": best_epoch,
                "global_step": best_step,
                "best_val_soft_acc": best_val_soft,
                "best_val_hard_acc": best_val_hard,
                "best_val_gap": best_val_gap,
                "config": cfg,
                "config_path": str(cfg_path),
                "connectivity_seed": conn_seed,
                "meta": {
                    "family": family,
                    "size": mcfg.get("size"),
                    "profile_source": mcfg.get("profile_source"),
                    "dataset": mcfg.get("dataset", dcfg.get("dataset")),
                    "teacher_enabled": bool(teacher is not None),
                },
            }
            torch.save(ckpt, best_path)
            print(f"  saved best checkpoint val_soft={best_val_soft:.4f} -> {best_path}")

        # 重置累计的训练损失与准确率统计，准备开始新一个epoch的训练
        running_loss = 0.0
        running_correct = 0
        running_total = 0

    with (
        open(train_step_log_path, "w", newline="", encoding="utf-8")
        if log_step_loss
        else nullcontext()
    ) as step_log_f:
        step_log_csv = csv.writer(step_log_f) if step_log_f is not None else None
        if step_log_csv is not None:
            step_log_csv.writerow(["global_step", "epoch", "train_loss", "lr"])

        # 对每个epoch和batch进行循环，前向传播、计算损失（支持普通交叉熵和带有teacher的知识蒸馏KL损失）、反向传播和梯度更新；
        for epoch in range(1, max_epochs + 1):
            model.train()
            teacher_enabled = teacher is not None
            epoch_train_loss_sum = 0.0
            epoch_train_samples = 0
            for batch in train_loader:
                global_step += 1
                x_teacher = None
                if teacher_enabled:
                    (x, x_teacher), y = batch
                else:
                    x, y = batch
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if teacher_enabled and x_teacher is not None:
                    x_teacher = x_teacher.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                out = model(x)
                logits = compute_class_scores(out, num_classes, tau, beta)
                ce = criterion(logits, y) # 计算交叉熵损失
                if teacher_enabled:
                    with torch.no_grad():
                        if x_teacher is None:
                            raise RuntimeError("teacher is enabled but x_teacher is missing from train loader")
                        t_logits = teacher(x_teacher)
                    if t_logits.shape != logits.shape:
                        raise ValueError(
                            f"Teacher logits shape {tuple(t_logits.shape)} must match "
                            f"student logits shape {tuple(logits.shape)}"
                        )
                    kd = F.kl_div(
                        F.log_softmax(logits / kd_temperature, dim=-1),
                        F.softmax(t_logits / kd_temperature, dim=-1),
                        reduction="batchmean",
                    ) * (kd_temperature ** 2) # 计算KL散度损失，KL散度衡量两个概率分布之间的差异
                    loss = (1.0 - teacher_weight) * ce + teacher_weight * kd # 计算蒸馏损失
                else:
                    loss = ce # 计算交叉熵损失

                loss.backward()
                optimizer.step() # 更新模型参数

                bs = x.size(0)
                epoch_train_loss_sum += float(loss.item()) * bs
                epoch_train_samples += int(bs)

                if step_log_csv is not None and (global_step % step_loss_log_interval == 0):
                    lr_now = float(optimizer.param_groups[0]["lr"])
                    step_log_csv.writerow(
                        [
                            global_step,
                            epoch,
                            f"{float(loss.detach().cpu()):.8f}",
                            f"{lr_now:.12f}",
                        ]
                    )
                    if step_loss_log_flush_every == 0 or global_step % max(
                        step_loss_log_flush_every, 1
                    ) == 0:
                        step_log_f.flush()

                running_loss += float(loss.item()) * bs
                pred = logits.argmax(dim=-1)
                running_correct += int((pred == y).sum().item())
                running_total += int(y.numel())

                if global_step % log_every_steps == 0:
                    print(
                        f"epoch={epoch}/{max_epochs} step={global_step} "
                        f"train_loss_running={running_loss / max(running_total, 1):.4f}"
                    )

                if global_step % eval_every_steps == 0:
                    run_validation(epoch, global_step)

            mean_epoch_loss = epoch_train_loss_sum / max(epoch_train_samples, 1)
            with open(train_epoch_log_path, "a", newline="", encoding="utf-8") as fcsv:
                w = csv.writer(fcsv)
                w.writerow([epoch, f"{mean_epoch_loss:.8f}"])

        # Flush the tail segment if the final step does not hit eval interval.
        if running_total > 0:
            run_validation(max_epochs, global_step)

    last_ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(max_epochs),
        "max_epochs": int(max_epochs),
        "global_step": int(global_step),
        "best_val_soft_acc": float(best_val_soft),
        "best_val_hard_acc": float(best_val_hard),
        "best_val_gap": float(best_val_gap),
        "best_epoch": int(best_epoch),
        "best_step": int(best_step),
        "config": cfg,
        "config_path": str(cfg_path),
        "connectivity_seed": conn_seed,
        "meta": {
            "family": family,
            "size": mcfg.get("size"),
            "profile_source": mcfg.get("profile_source"),
            "dataset": mcfg.get("dataset", dcfg.get("dataset")),
            "teacher_enabled": bool(teacher is not None),
        },
    }
    torch.save(last_ckpt, last_path)
    print(f"saved last checkpoint -> {last_path}")

    if not best_path.exists():
        raise RuntimeError("Training finished without producing a best checkpoint.")

    best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False) # 加载最佳模型
    best_model = build_soft_model_from_config(mcfg, connectivity_seed=conn_seed).to(device) # 构建最佳模型
    best_model.load_state_dict(best_ckpt["model_state_dict"], strict=True) # 加载最佳模型权重
    best_hard = build_hard_model_from_soft(best_model, mcfg).to(device) # 构建最佳硬模型    

    test_soft_acc, test_hard_acc, test_gap = evaluate_soft_hard_pair(
        best_model, best_hard, test_loader, device, num_classes, tau, beta
    ) # 评估最佳模型在测试集上的软硬一致性指标

    final_test = {
        "best_epoch": int(best_ckpt.get("epoch", best_epoch)),
        "best_global_step": int(best_ckpt.get("global_step", best_step)),
        "best_val_soft_acc": float(best_ckpt.get("best_val_soft_acc", best_val_soft)),
        "best_val_hard_acc": float(best_ckpt.get("best_val_hard_acc", best_val_hard)),
        "best_val_gap": float(best_ckpt.get("best_val_gap", best_val_gap)),
        "test_soft_acc": float(test_soft_acc),
        "test_hard_acc": float(test_hard_acc),
        "test_gap": float(test_gap),
    }
    with open(final_test_path, "w", encoding="utf-8") as f:
        json.dump(final_test, f, indent=2)

    summary = {
        "run_dir": str(run_dir),
        "family": family,
        "profile_source": mcfg.get("profile_source"),
        "size": mcfg.get("size"),
        "train_log_csv": str(train_log_path),
        "train_epoch_log_csv": str(train_epoch_log_path),
        "layer_stats_csv": str(layer_stats_path),
        "final_test_json": str(final_test_path),
        "checkpoint_dir": str(run_ckpt_dir),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }
    if log_step_loss:
        summary["train_step_log_csv"] = str(train_step_log_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Final test from best checkpoint:", final_test)
    print("Done.", summary)


if __name__ == "__main__":
    main()
