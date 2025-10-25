[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/vision/stable/index.html)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-000000?logo=openai&logoColor=white)](https://gymnasium.farama.org/)
[![Procgen](https://img.shields.io/badge/Procgen-3776AB?logo=python&logoColor=white)](https://github.com/openai/procgen)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![opencv-python](https://img.shields.io/badge/opencv--python-5C3EE8?logo=opencv&logoColor=white)](https://pypi.org/project/opencv-python/)


# Generalization in Procedurally Generated Environments

This repository contains code and instructions for the project "Generalization in Procedurally Generated Environments" as part of the Deep Reinforcement Learning course at the University of Zurich (UZH).

___

## ToDo
- Remove everything but SiT and baseline (cnn or any NN)
- Try to make the SiT smaller, training must become much faster
- Try to replace SiT with a smaller equivariant architecture (e.g. G-CNN)
- Test to run this on the HuggingFace hub
- Improve logging massively to track experiments and results
- Start Latex documentation (lots of sources to add, background of SiT, ...)
- clean up code, remove unnecessary comments, make documentation nicer, ...

Week 21.10.2025
- all environments work on the invididual machines
- Jan: starts report on latec
- Sohit: SiT understand, read, cleanup, try to make it smaller
- Luca: run in on hugging face && cleanup code, remove unused code, set default values we never change, only keep models and algos we plan to use
- Zusan: Data augmentation method: edge filters, remove noise in the images, run them to be sure they work, think about very clever ones
- Sandrin: intermediate report, create draft of project scope, goals, tasks, timeline


---

## Original Task Description
Train on subsets of generated levels and test on unseen ones to measure true generalization. Explore data augmentation
and inductive biases that improve transfer.

### Tasks
1) Familiarize with a procedural environment like Minigrid [[1]](https://minigrid.farama.org/), ProcGen [[2]](https://github.com/openai/procgen), or, if ambitious, Nethack [[3]](https://github.com/facebookresearch/nle)
2) Train an agent using an RL algorithm like PPO or SAC with an architecture appropriate to the environment (e.g., CNN or
the supposedly especially generalizable Symmetry-Invariant Vision Transformer [[4]](https://openreview.net/attachment?id=SWrwurHAeq&name=pdf) (~70k params)).
3) Evaluate the agent’s generalization performance by training on some levels, testing on others.
4) Experiment with techniques that aim to improve generalization such as data augmentation or regularization methods (e.g.,
dropout, batch normalization) and analyze their impact.

___

## Prequisites
- You need python 3.10 otherwise procgen and gym will not run!

___

## Setup

```bash
git clone https://github.com/Altishofer/deep-reinforcement-learning-uzh.git
```

```bash
cd deep-reinforcement-learning-uzh
```

```bash
python3.10 -m venv venv
```

```bash
# Linux / MacOS
source venv/bin/activate
```

```bash
# Windows
venv\Scripts\activate
```

```bash
pip install -r requirements_cpu.txt  --force-reinstall --no-cache-dir
```

___

## Run

```bash
python train2.py --env_name starpilot --device_id 0 --seed 1 --use_sit True --choice 0  --run_name Siet --num_mini_batch 96  --ppo_epoch 2  --hidden_size 64   
```

___

## References
[[1]](https://minigrid.farama.org/) Maxime Chevalier-Boisvert et al.: “Minigrid & Miniworld: Modular & Customizable Reinforcement Learning Environments
for Goal-Oriented Tasks.” NeurIPS (2023).

[[2]](https://github.com/openai/procgen) Karl Cobbe et al.: “Leveraging Procedural Generation to Benchmark Reinforcement Learning.” ArXiv preprint (2019),
available at arXiv:1912:01588.

[[3]](https://github.com/facebookresearch/nle) Heinrich Küttler et al.: “The NetHack Learning Environment” NeurIPS (2020). Available
at .

[[4]](https://openreview.net/attachment?id=SWrwurHAeq&name=pdf) Matthias Weissenbacher et al.: “SiT: Symmetry-Invariant Transformers for Generalisation in Reinforcement Learning.”
ArXiv preprint (2024), available at arXiv:2406:15025.


## GitHub Resources
- [ProcGen Repository](https://github.com/openai/procgen?tab=readme-ov-file)
- [Minigrid Repository](https://github.com/Farama-Foundation/Minigrid)
- [Gym OpenAI](https://github.com/openai/gym)
- [Rotation Equivariant Vision Transformers](https://github.com/matthias-weissenbacher/SiT)








1) data is augmented by random shifts, rotations, and flips
2) architecture is a symmetry-invariant vision transformer (SiT) that is equivariant to
3) train on PPO variant (DrAco) that uses data augmentation

0) new data augmentation techniques
1) small Sit
2) our own PPO implementation
