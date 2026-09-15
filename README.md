# Convolutional Differentiable Logic Gate Networks (Pure PyTorch)

本仓库仅保留 `logictreenet`（卷积逻辑门树）实现，用于复现
**Convolutional Differentiable Logic Gate Networks**。

训练、评估、离散化入口保持不变：

- `python src/train.py --config ...`
- `python src/evaluate.py --checkpoint ...`
- `python src/discretize.py --checkpoint ...`

## 1. 当前复现范围

已落地内容：

- 8 个论文档位配置（Table-6 口径）：
  - CIFAR-10: `S/M/B/L/G`
  - MNIST: `S/M/L`
- 卷积逻辑树层（`d=3`，8 leaves，7 gates）与硬离散推理对照实现。
- OR pooling（max t-conorm，对应 `max_pool2d`）。
- residual gate init（默认 `z_A=5`，其余 0）。
- 训练流程支持按 step 周期验证与 best checkpoint 保存。
- CIFAR `B/L/G` 支持 teacher 蒸馏接口（外部 checkpoint）。
- CIFAR 输入模式：`small`（9 通道阈值二值）、`large_threshold31`（93 通道）、`large_bitplane5`（15 通道 bit-plane）。

未复现内容：

- 论文 fused CUDA 后端和其速度表现。

## 1.1 与之前线性逻辑门网络的实现区别（简述）

- **输入形态不同**：之前线性网络用展平向量输入；当前卷积网络在代码里保留 `BCHW` 图像张量，并在 `src/data/cifar10.py` 中按 `data.cifar10_input_mode` 构造多通道特征（`small` / `large_threshold31` / `large_bitplane5`）。
- **主干结构不同**：之前是“全连接随机二入逻辑层”堆叠；当前是 `ConvLogicTreeLayer + OrPool2d` 的卷积逻辑树主干，再接 3 层随机连接逻辑 head。
- **离散化对象不同**：之前离散化主要保存 dense 层的 `connectivity + gate_ids`；当前额外保存卷积树的 `conv_leaf_indices + conv_gate_ids`，用于硬推理重建。
- **训练配置不同**：之前主要手工配置单模型；当前通过 `configs/paper_table6/*.yaml` 选择 8 个档位，并支持 CIFAR `B/L/G` 的 teacher 蒸馏字段。

## 2. 目录结构（关键部分）

```text
configs/
  paper_table6/
    cifar10_s.yaml
    cifar10_m.yaml
    cifar10_b.yaml
    cifar10_l.yaml
    cifar10_g.yaml
    mnist_s.yaml
    mnist_m.yaml
    mnist_l.yaml

src/
  train.py
  evaluate.py
  discretize.py
  config_profiles/paper_table6.py
  data/
    loaders.py
    mnist.py
    cifar10.py
  models/
    model_factory.py
    conv_logic_tree_layer.py
    hard_conv_logic_tree_layer.py
    logic_tree_net.py
    hard_logic_tree_net.py
    or_pool.py
```

## 3. 环境安装 convlogicgate

推荐先装 PyTorch，再装其余依赖。示例（CUDA 12.4）：

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

CPU 环境可直接：

```bash
pip install -r requirements.txt
```

## 4. 快速开始

### 4.1 训练（8 档模型）

```bash
# CIFAR-10
python src/train.py --config configs/paper_table6/cifar10_s.yaml
python src/train.py --config configs/paper_table6/cifar10_m.yaml
python src/train.py --config configs/paper_table6/cifar10_b.yaml
python src/train.py --config configs/paper_table6/cifar10_l.yaml
python src/train.py --config configs/paper_table6/cifar10_g.yaml

# MNIST
python src/train.py --config configs/paper_table6/mnist_s.yaml
python src/train.py --config configs/paper_table6/mnist_m.yaml
python src/train.py --config configs/paper_table6/mnist_l.yaml
```

### 4.2 评估与离散化

```bash
python src/evaluate.py --checkpoint logs/cifar10_s/run_<timestamp>/checkpoints/best_model.pt
python src/discretize.py --checkpoint logs/cifar10_s/run_<timestamp>/checkpoints/best_model.pt
```

## 5. 配置接口（Config API）

### 5.1 统一字段

- `model.family`: `logictreenet`
- `model.dataset`: `mnist | cifar10`
- `model.size`: `S | M | B | L | G`（MNIST 仅 `S/M/L`）
- `model.profile_source`: `paper_table6`
- `output.tau`, `output.beta`
- `checkpoint.seed`（连接性等随机种子）。`checkpoint.save_dir` 若仍存在 YAML 中会被**忽略**：学生模型 checkpoint 固定写入本次运行的 `logs/<dataset>_<size>/run_<timestamp>/checkpoints/`。

### 5.2 `logictreenet` 关键字段

- `model.k`: 主通道规模（profile 会自动填充）
- `model.output_gate_factor`: `ox`（profile 会自动填充）
- `model.receptive_field_size`（默认 3）
- `ConvLogicTreeLayer` 明确约束：每棵树固定只观察 `2` 个输入通道（`channels_per_tree=2`），8 个叶子仅从这 2 个通道对应的局部感受野采样。
- `model.gate_init`（默认 `residual_a`）
- `training.eval_every_steps`（默认：CIFAR `2000`，MNIST `500`；可在 YAML 中覆盖）
- `training.log_every_steps`（默认 `100`）
- `training.optimizer`: 默认 `adamw`
- `training.weight_decay`: profile 自动填充

### 5.3 CIFAR 输入模式

- `data.cifar10_input_mode`（仅下列三选一）：
  - `small`：3×3 阈值二值化，**9** 通道（Table-6 中 S/M 默认）。
  - `large_threshold31`：3×31 阈值二值化，**93** 通道（Table-6 中 B/L/G 默认）。
  - `large_bitplane5`：RGB 各 5 bit-plane，**15** 通道。

## 6. Teacher 蒸馏（CIFAR B/L/G）

在 `cifar10_b.yaml / cifar10_l.yaml / cifar10_g.yaml` 中配置：

- `teacher.enabled`: `true/false`
- `teacher.checkpoint`: 路径
- `teacher.arch`: 可选；若填写则按 torchvision 架构实例化再加载 state_dict
- `teacher.distill_weight`: 蒸馏损失权重，默认 `0.5`
- `teacher.kd_temperature`: KL 蒸馏温度（teacher/student logits softmax 温度），默认 `sqrt(2)`
- `teacher.temperature_scale`: `teacher.kd_temperature` 的兼容别名（例如 `teacher.temperature_scale: sqrt(2)`）

与输出层温度区分：

- `output.tau`: LogicTreeNet 的 group-sum class score 缩放温度（用于 `compute_class_scores`）。
- `teacher.kd_temperature`: 仅用于蒸馏 KL，不影响 `output.tau`。

说明：

- 若 `teacher.arch` 为空，`teacher.checkpoint` 需能直接 `torch.load` 成 `nn.Module`。
- 若 `teacher.arch` 非空，checkpoint 需提供可加载的 state dict（`model_state_dict` / `state_dict` / 纯 dict）。

## 7. 训练产物

每次运行在 `logs/run_<timestamp>/` 下生成：

- `train_log.csv`: 按验证间隔记录训练/验证指标（含 `epoch`、`global_step`）
- `train_epoch_log.csv`: **每个 epoch 一行**，该 epoch 内按样本加权的平均训练 loss（列 `epoch`, `train_loss`）；`training_loss.png` 依赖此文件
- `train_step_log.csv`（可选，`training.log_step_loss`）：按 `step_loss_log_interval`（默认每 **100** 个 `global_step`）稀疏记录 `global_step`, `epoch`, `train_loss`, `lr`；`step_loss_log_flush_every` 仍为刷盘频率
- `layer_stats.csv`: 每次验证时按层记录门控统计（含 `epoch`）
- `final_test.json`: best checkpoint 对应测试集 soft/hard 准确率
- `run_summary.json`: 本次运行摘要（含 `checkpoint_dir`、`best_checkpoint`、`last_checkpoint` 等路径）

可视化（依赖 `matplotlib`；横轴为 **epoch**，仅绘制 CSV 中的真实记录点；详见 `scripts/plot_training_run.py` 文件头）：

```bash
python scripts/plot_training_run.py --run-dir logs/cifar10_b/run_20260511_180518
```

默认在同一目录生成 `training_loss.png`（`train_epoch_log.csv`）、`val_acc.png`（`train_log.csv` 中每次验证一行）、可选 `layer_stats.png`（`layer_stats.csv`）。可用 `--epoch-log` / `--layer-stats` / `--out` 覆盖路径。

在同一 run 目录下的 `**checkpoints/**` 中生成（与 CSV 日志同根，互不覆盖）：

- `best_model.pt`：验证集 soft acc 创新高时写入（字段与原先一致）
- `last_model.pt`：训练结束时的最终权重与优化器状态等

离散化命令会额外输出：

- `*_discrete.pt`

`logictreenet` 离散包包含：

- `conv_leaf_indices`
- `conv_gate_ids`
- `head_connectivities`
- `head_gate_ids`
- `meta`

## 8. 仓库裁剪说明

- 已移除旧版 flat baseline 配置与对应训练分支。
- 当前仅支持 `configs/paper_table6/` 下 8 个模型配置。
- `train/evaluate/discretize` 都只保留 `logictreenet` 路径。

## 9. 静态检查与测试

本地可先做语法检查：

```bash
python -m compileall src tests
```

完整测试：

```bash
python -m pytest tests -v
```

新增测试覆盖：

- ConvLogicTree 输出 shape 与参数 shape
- OrPool 与 max-pool 等价
- residual init 概率分布
- conv soft/hard 一致性
- Table-6 profile 映射
- Logictree 数据模式输入通道校验

## 10. 论文口径说明

- profile 数值按你指定的 Table-6 口径固化（含 CIFAR-G `k=2560`）。
- CIFAR 大输入（93 通道）对应论文中的多阈值二值化管线；`large_bitplane5` 为可选替代输入，需在配置中显式指定。
- 本仓库目标是结构与训练策略复现，不追求 fused CUDA 速度复现。

