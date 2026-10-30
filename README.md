# Conv-DLGN: Convolutional Differentiable Logic-Gate Networks

**English** | [简体中文](README.zh-CN.md)

**Spatial feature learning with exportable logic trees.**

An independent PyTorch implementation and reproduction study of [Convolutional Differentiable Logic Gate Networks](https://arxiv.org/abs/2411.04732). The contributions below describe the implementation and evaluation delivered by this repository; convolutional logic trees, OR pooling, and residual initialization follow the original method.

## Abstract

Image classification with logic gates calls for local feature extraction as well as a path from differentiable training to discrete execution. Conv-DLGN implements a spatial hierarchy of convolutional logic trees, OR pooling, and a sparse logic classification head. Trainable operator mixtures support gradient optimization, while hard export assigns a Boolean operator to every gate without changing the stored connectivity. On a representative CIFAR-10 B configuration, the implementation achieves **70.32% soft and 70.27% hard test accuracy**, a **0.05 percentage-point** difference. This result demonstrates accurate classification by the exported logic-tree model and close agreement between its two execution modes.

## Contributions

- **Spatially structured logic computation.** Local depth-3 logic trees and OR pooling preserve image feature maps through a convolutional hierarchy.
- **A complete soft-to-hard workflow.** Operator learning, residual gate initialization, validation-based checkpoint selection, and discrete export are integrated in PyTorch.
- **Direct evidence for exported inference.** Paired MNIST and CIFAR-10 measurements assess the hard model itself, including 70.27% hard accuracy on the reported CIFAR-10 B configuration.

## Methodology

![Conv-DLGN uses image encoding, local logic-tree convolution, OR pooling, a sparse logic head and class-group scores. Each depth-3 tree has eight leaves and seven gates.](docs/figures/overview.svg)

*Figure 1. Local logic trees operate across image locations. Their outputs form spatial feature maps, which are pooled and passed to a sparse logic head. The inset shows the eight-leaf, seven-gate tree used by each output channel.*

**Local composition.** Threshold or bit-plane encoding produces the input channels. A convolutional tree samples eight fixed leaves from a local receptive field, using up to two input channels, and combines them through seven two-input gates arranged in three levels. The same learned tree is applied across spatial positions. OR pooling uses a max t-conorm; class-group sums aggregate the logic head outputs.

**Learned operators, discrete export.** Each gate learns a softmax-weighted mixture of 16 Boolean operators. Residual initialization favors the identity operator. Export replaces each mixture with its argmax operator while preserving the tree leaves and sparse head connections. See [convolutional trees](src/models/conv_logic_tree_layer.py), [network composition](src/models/logic_tree_net.py), and [hard model](src/models/hard_logic_tree_net.py).

## Experiments

**Question: can a spatial logic-tree classifier preserve its classification performance after gate discretization?** Table 1 pairs soft and hard evaluation within each configuration.

| Dataset / configuration | Soft test accuracy | Hard test accuracy | Soft − hard (pp) |
| --- | ---: | ---: | ---: |
| MNIST / S | 98.49% | 97.55% | 0.94 |
| CIFAR-10 / S | 58.93% | 57.20% | 1.73 |
| CIFAR-10 / B | **70.32%** | **70.27%** | **0.05** |

*Table 1. Representative completed local configurations; pp denotes percentage points. Each row compares the same selected model before and after gate export. Profiles and training settings differ between rows.*

**Takeaway.** The CIFAR-10 B result is the central demonstration: its exported model reaches 70.27% accuracy while remaining within 0.05 pp of soft execution. MNIST S and CIFAR-10 S establish additional operating points for the same implementation path. Cross-row differences combine model and training choices, so the table is interpreted as paired conversion evidence rather than a controlled scaling ablation.

**Protocol.** Validation soft accuracy selects checkpoints before test evaluation. The entries are individual completed configurations, not averages across independent seeds. Input encoding and teacher use are configuration-specific; teacher-enabled B/L/G configurations require their external teacher checkpoint during training. The repository provides PyTorch classification and export measurements; throughput and energy require a separately measured execution backend. Datasets, checkpoints, and raw experiment histories are maintained outside the repository.

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Full training runs.
python src/train.py --config configs/paper_table6/mnist_s.yaml
python src/train.py --config configs/paper_table6/cifar10_s.yaml
```

The examples start new S-profile runs. Evaluate or export the checkpoint produced by your run:

```bash
python src/evaluate.py --checkpoint path/to/checkpoint.pt
python src/discretize.py --checkpoint path/to/checkpoint.pt
python -m pytest tests -v
```

Checkpoints are saved under each run directory. The table summarizes previously completed configurations; these example commands do not specify the complete historical training environment.

## Repository guide

| Entry | Purpose |
| --- | --- |
| [configs/paper_table6/](configs/paper_table6/) | MNIST S/M/L and CIFAR-10 S/M/B/L/G profiles; experiment evidence is summarized in Table 1 |
| [src/data/](src/data/) | Threshold and bit-plane input encodings |
| [src/models/](src/models/) | Soft/hard tree layers, OR pooling and classification heads |
| [MODEL_LAYER_SUMMARY.md](MODEL_LAYER_SUMMARY.md) | Chinese layer guide |
| [REFERENCE_NOTES.md](REFERENCE_NOTES.md) | Attribution and implementation scope |

Related projects: [DLGN](https://github.com/13710833172/DLGN) provides flattened sparse logic networks; [Logic-SNN](https://github.com/13710833172/Logic-SNN) adds spiking temporal dynamics.

## Conclusion

Conv-DLGN connects spatial feature extraction with discrete logic-tree execution. Its reported CIFAR-10 B model makes that connection concrete: **70.27% hard accuracy with a 0.05 pp soft-to-hard difference**.
