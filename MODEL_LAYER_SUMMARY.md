# 8 档 LogicTreeNet 结构与形状（论文 Appendix A.1 + Table 6）

本文档与当前实现 [`src/models/logic_tree_net.py`](src/models/logic_tree_net.py) 及 [`src/config_profiles/paper_table6.py`](src/config_profiles/paper_table6.py) 对齐，覆盖 `paper_table6` 的 8 个模型：

- CIFAR-10：`S / M / B / L / G`
- MNIST：`S / M / L`

约定：

- 下文用 `B` 表示 batch size。
- 分类数均为 `10`。
- `ox` = `output_gate_factor`（Table 6 中的 `ox`）。
- 每个卷积块：`ConvLogicTreeLayer` + `OrPool2d(kernel=2, stride=2)`（OR pooling，实现为 `max_pool2d`）。
- 最终 logits：`compute_class_scores` 对最后一维 `320*k*ox` 做 GroupSum（按类分组 **sum**，再除以 `tau`）。

形状自动化校验见 [`tests/test_paper_table6_shapes.py`](tests/test_paper_table6_shapes.py)。

---

## Table 6 超参（与配置一致）

| 档位 | 数据集 | k | ox | τ | lr | weight_decay | batch_size |
|------|--------|---|----|---|-----|----------------|------------|
| S | CIFAR-10 | 32 | 1 | 20.0 | 0.02 | 0.002 | 128 |
| M | CIFAR-10 | 256 | 1 | 40.0 | 0.02 | 0.002 | 128 |
| B | CIFAR-10 | 512 | 2 | 280.0 | 0.02 | 0.002 | 128 |
| L | CIFAR-10 | 1024 | 2 | 340.0 | 0.02 | 0.002 | 128 |
| G | CIFAR-10 | 2560 | 1 | 450.0 | 0.02 | 0.001 | 128 |
| S | MNIST | 16 | 2 | 6.5 | 0.01 | 0.0 | 512 |
| M | MNIST | 64 | 2 | 28.0 | 0.01 | 0.0 | 256 |
| L | MNIST | 256 | 1 | 35.0 | 0.01 | 0.0 | 128 |

---

## 一、CIFAR-10：Backbone（Appendix A.1）

- **输入**：`[B, C_in, 32, 32]`  
  - S/M：`C_in = 9`（`data.cifar10_input_mode: small`）  
  - B/L/G 默认：`C_in = 93`（`large_threshold31`）；若配置为 `large_bitplane5` 则 `C_in = 15`

- **Block1**：`out_channels = k`，`rf=3`，`padding=1` → `OrPool2d` → `[B, k, 16, 16]`
- **Block2**：`out_channels = 4k`，`rf=3`，`padding=1` → `OrPool2d` → `[B, 4k, 8, 8]`
- **Block3**：`out_channels = 16k`，`rf=3`，`padding=1` → `OrPool2d` → `[B, 16k, 4, 4]`
- **Block4**：`out_channels = 32k`，`rf=3`，`padding=1` → `OrPool2d` → `[B, 32k, 2, 2]`

- **Flatten**：`[B, 32k, 2, 2] → [B, 128k]`  
  - `backbone_feature_dim = 128k`（与真实 flatten 一致）

### Head（CIFAR 与 MNIST 相同公式）

记 `ox = output_gate_factor`，`output_neuron_factor = 320`（代码默认，对应论文中 `320*k*ox` 的基准）：

- **Head1**：`128k → 1280·k·ox`
- **Head2**：`1280·k·ox → 640·k·ox`
- **Head3**：`640·k·ox → 320·k·ox`
- **GroupSum**：`320·k·ox` 维按 10 类分组求和并除以 `τ` → `[B, 10]`

属性 `model.head_width` 为最后一层宽度，即 **`320*k*ox`**；每类神经元数为 **`32*k*ox`**。

---

### CIFAR-10-S

- `k=32`, `ox=1`, `C_in=9`
- Pool 后：`[B,32,16,16] → [B,128,8,8] → [B,512,4,4] → [B,1024,2,2]`
- Flatten：`[B, 4096]`
- Head：`4096 → 40960 → 20480 → 10240`；logits：`[B,10]`（每类 1024）

### CIFAR-10-M

- `k=256`, `ox=1`, `C_in=9`
- Pool 后：`[B,256,16,16] → [B,1024,8,8] → [B,4096,4,4] → [B,8192,2,2]`
- Flatten：`[B, 32768]`
- Head：`32768 → 327680 → 163840 → 81920`；logits：`[B,10]`（每类 8192）

### CIFAR-10-B

- `k=512`, `ox=2`, `C_in=93`
- Pool 后：`[B,512,16,16] → [B,2048,8,8] → [B,8192,4,4] → [B,16384,2,2]`
- Flatten：`[B, 65536]`
- Head：`65536 → 1310720 → 655360 → 327680`；logits：`[B,10]`（每类 32768）

### CIFAR-10-L

- `k=1024`, `ox=2`, `C_in=93`
- Pool 后：`[B,1024,16,16] → [B,4096,8,8] → [B,16384,4,4] → [B,32768,2,2]`
- Flatten：`[B, 131072]`
- Head：`131072 → 2621440 → 1310720 → 655360`；logits：`[B,10]`（每类 65536）

### CIFAR-10-G

- `k=2560`, `ox=1`, `C_in=93`
- Pool 后：`[B,2560,16,16] → [B,10240,8,8] → [B,40960,4,4] → [B,81920,2,2]`
- Flatten：`[B, 327680]`
- Head：`327680 → 3276800 → 1638400 → 819200`；logits：`[B,10]`（每类 81920）

---

## 二、MNIST：Backbone（Appendix A.1）

- **输入**：`[B, 1, 28, 28]`（二值）

- **Block1**：`out_channels = k`，`rf=5`，`padding=0` → `OrPool2d` → `[B, k, 12, 12]`
- **Block2**：`out_channels = 3k`，`rf=3`，`padding=1` → `OrPool2d` → `[B, 3k, 6, 6]`
- **Block3**：`out_channels = 9k`，`rf=3`，`padding=1` → `OrPool2d` → `[B, 9k, 3, 3]`

- **Flatten**：`[B, 9k, 3, 3] → [B, 81k]`  
  - `backbone_feature_dim = 81k`（与真实 flatten 一致）

Head 公式与 CIFAR 相同（`1280·k·ox → 640·k·ox → 320·k·ox`）。

---

### MNIST-S

- `k=16`, `ox=2`
- Pool 后：`[B,16,12,12] → [B,48,6,6] → [B,144,3,3]`
- Flatten：`[B, 1296]`
- Head：`1296 → 40960 → 20480 → 10240`；logits：`[B,10]`（每类 1024）

### MNIST-M

- `k=64`, `ox=2`
- Pool 后：`[B,64,12,12] → [B,192,6,6] → [B,576,3,3]`
- Flatten：`[B, 5184]`
- Head：`5184 → 163840 → 81920 → 40960`；logits：`[B,10]`（每类 4096）

### MNIST-L

- `k=256`, `ox=1`
- Pool 后：`[B,256,12,12] → [B,768,6,6] → [B,2304,3,3]`
- Flatten：`[B, 20736]`
- Head：`20736 → 327680 → 163840 → 81920`；logits：`[B,10]`（每类 8192）

---

## 三、资源说明（不改论文结构）

CIFAR **B / L / G** 在纯 PyTorch 下参数量与激活体积极大（尤其 L/G 的 Head1 宽度在百万级以上），显存与单次 step 时间会显著高于 S/M。这是论文设定下的预期成本；若 OOM，应从硬件、梯度检查点、工程批处理等外围手段缓解，**不应**为跑通而擅自缩小 `k`、head 或 backbone 通道 schedule。

