[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/vision/stable/index.html)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-000000?logo=openai&logoColor=white)](https://gymnasium.farama.org/)
[![Procgen](https://img.shields.io/badge/Procgen-3776AB?logo=python&logoColor=white)](https://github.com/openai/procgen)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![opencv-python](https://img.shields.io/badge/opencv--python-5C3EE8?logo=opencv&logoColor=white)](https://pypi.org/project/opencv-python/)

# Generalization in Procedurally Generated Environments

This repository contains the official implementation accompanying the project **“Generalization in Procedurally Generated Environments”**.  
We study how reinforcement learning algorithms generalize when trained on small subsets of Procgen levels and tested on large unseen level distributions. The codebase provides reproducible training pipelines, unified evaluation tools, and implementations of regularization and augmentation strategies.

---

## 1. Overview

Procedurally generated environments provide a principled benchmark for true generalization, separating memorization from robust policy learning. This work evaluates generalization under **limited training diversity** (50 levels) across two Procgen tasks:

- **fruitbot**
- **chaser**

We revive and unify two architectures with reported strong generalization:

- **SiT** (Symmetry‑Invariant Vision Transformer)
- **IMPooLA** (lightweight CNN with improved pooling)

and compare them against several RL baselines:

- PPO (existing)
- DQN (existing)
- GRPO (new)
- V‑Trace (new)
- PPO with Random Network Distillation (new)

Additionally, we evaluate the effect of **DrAC**, **input augmentation**, and **dropout** on generalization performance.


---

## 4. Installation

**Python 3.10 is required.**

```bash
git clone https://github.com/LucaZwahlen/RL25-project-19.git
cd RL25-project-19
python3.10 -m venv venv
source venv/bin/activate     # Linux/Mac
# venv\Scripts\activate      # Windows
python install.py
```

---

## 5. Training

### IMPooLA / V‑Trace

```bash
python impoola_cnn/ppo_training.py --env_name fruitbot
```

### SiT (example configuration)

```bash
python train2.py --env_name fruitbot --device_id 0 --seed 1 --use_sit True \
  --choice 0 --run_name SiT --num_mini_batch 96 --ppo_epoch 2 --hidden_size 64
```

---

## 6. Summary of Results

### Fruitbot

| Algorithm | Test Reward | GG | Success | SPL | Time |
|----------|-------------|----|---------|-----|------|
| DQN | 4.09 | 8.73 | 0% | 0% | 01:49:42 |
| GRPO | 29.23 | 1.09 | 84% | 16% | 00:36:11 |
| PPO | 29.29 | 0.80 | 85% | 18% | 00:34:03 |
| PPO-RND | 20.25 | 0.12 | 50% | 0% | 01:06:52 |
| **V-Trace** | **29.96** | **0.03** | **89%** | **20%** | **00:20:53** |

### Chaser

| Algorithm | Test Reward | GG | Success | SPL | Time |
|----------|-------------|----|---------|-----|------|
| DQN | 1.89 | 0.77 | 3% | 0% | 02:20:19 |
| GRPO | 10.81 | 0.31 | 78% | 16% | **00:18:41** |
| **PPO** | **11.54** | 1.22 | **84%** | **38%** | 00:56:12 |
| PPO-RND | 11.28 | 1.27 | 83% | 28% | 01:05:21 |
| V‑Trace | 10.83 | **0.13** | 78% | 29% | 00:29:26 |

---

## 7. Key Findings

- **V‑Trace** provides the most stable learning and smallest generalization gap.
- **PPO** and **GRPO** reach strong performance but overfit more strongly.
- **DQN** fails on both tasks due to visual complexity.
- **Dropout** (p = 0.01) offers a mild improvement.
- **Augmentation** improves generalization but hurts reward.
- **DrAC** significantly slows training and adds variance without clear benefits.

---

## 8. Contributors

- **Luca Zwahlen** 
- **Sandrin Hunkeler**
- **Sohith Vishnu Sai Yachamaneni**
- **Jan Zurbrügg**
- **Claudia Jurado Santos**

---

## 9. References

- Minigrid: https://minigrid.farama.org/
- Procgen: https://github.com/openai/procgen
- NLE: https://github.com/facebookresearch/nle
- SiT: https://openreview.net/attachment?id=SWrwurHAeq&name=pdf

---
