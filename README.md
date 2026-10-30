# Conv-DLGN: Convolutional Logic-Gate Networks

A pure-PyTorch implementation of convolutional logic-tree networks for MNIST and CIFAR-10, following [Convolutional Differentiable Logic Gate Networks](https://arxiv.org/abs/2411.04732). This repository is an implementation/reproduction study, not the original authors' official release.

## Architecture

```text
Image -> threshold / bit-plane encoding -> local convolutional logic trees
      -> OR pooling -> sparse logic head -> class-group scores
```

Each depth-3 binary tree uses eight input leaves and seven logic gates, with leaves sampled from two channels in a local receptive field. Training uses operator mixtures and residual gate initialization; hard export selects one Boolean operator per gate.

Unlike the flattened [DLGN baseline](https://github.com/13710833172/DLGN), this model keeps spatial feature maps. Unlike [Logic-SNN](https://github.com/13710833172/Logic-SNN), it has no LIF temporal state.

## Recorded experiments

Representative completed configurations, summarized from local experiment results:

| Dataset / configuration | Soft test | Hard test |
| --- | ---: | ---: |
| MNIST S | 98.49% | 97.55% |
| CIFAR-10 S | 58.93% | 57.20% |
| CIFAR-10 B | 70.32% | 70.27% |

The configurations differ in training setup and are not a controlled scaling comparison or a multi-seed benchmark. These are local experimental observations; datasets, checkpoints, and experiment-history files are kept outside this repository. No new training or evaluation was performed for this documentation update.

The code contains eight model profiles: MNIST S/M/L and CIFAR-10 S/M/B/L/G. Profile availability does not mean every size has a completed experiment. The paper's fused CUDA backend, throughput, and full headline accuracy are not reproduced by this repository.

## Setup and usage

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Full training runs, not short smoke tests.
python src/train.py --config configs/paper_table6/mnist_s.yaml
python src/train.py --config configs/paper_table6/cifar10_s.yaml
```

Evaluate or export a newly trained checkpoint:

```bash
python src/evaluate.py --checkpoint logs/cifar10_s/<run>/checkpoints/best_model.pt
python src/discretize.py --checkpoint logs/cifar10_s/<run>/checkpoints/best_model.pt
```

Checkpoints are written inside each run directory. Weights, datasets, and experiment histories are not bundled with the source. Teacher-enabled B/L/G configurations additionally require the configured external teacher checkpoint.

```bash
python -m pytest tests -v
```

## Configuration and implementation notes

- CIFAR input modes: 9-channel thresholds, 93-channel thresholds, or 15-channel bit planes. Specify the intended mode explicitly.
- OR pooling uses a max t-conorm; hard gates use stored tree connectivity and operator IDs.
- Validation soft accuracy selects checkpoints; soft and hard test scores are reported together.
- Local PyTorch execution should not be used to infer optimized hardware latency or energy savings.

See [MODEL_LAYER_SUMMARY.md](MODEL_LAYER_SUMMARY.md) for the Chinese layer guide and [REFERENCE_NOTES.md](REFERENCE_NOTES.md) for attribution.
