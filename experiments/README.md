# Experiments

## Run the baselines (3x)

Every one of our baselines is run this way.

Note that for a valid SPL, the all-knowing paths must be computed (samples are existing in this repository)

To train our baselines, run:

```bash
  python -m experiments.run_ppo_baseline
  python -m experiments.run_grpo_baseline
  python -m experiments.run_ppo_rnd_baseline
  python -m experiments.run_vtrace_baseline
  python -m experiments.run_dqn_baseline
```

## Run the experiments

In addition to our baselines, we ran various experiments regarding dropout, augmentation, and DrAc.

Note that for a valid SPL, the all-knowing paths must be computed (samples are existing in this repository)

To train for our experiments, run:

```bash
  python -m experiments.run_experiments_vtrace_drac
  python -m experiments.run_experiments_vtrace_aug
```

Ther are other smaller experiments not mentioned anywhere, feel free to explore the [experiments](../experiments/) folder.

## Run the all-knowing path computation

This expects a well-trained agent checkpoint file and computes all 50'000 paths of the test set. To train PPO with additional information, run the following command with your desired environment and seed.

```powershell
python -m impoola_cnn.ppo_training --is_all_knowing --seed 1 --env_id chaser
```

To run the all-knowing path computation, run the following command, then rename and move the output files as neccessary (for example to: [eval/all_knowing_path_lengths](../impoola_cnn/impoola/eval/all_knowing_path_lengths/)):

```powershell
python -m experiments.compute_all_knowing_path --checkpoint /my/checkpoint/path
```

To change environment and seed here, change the hardcoded variables:

```python
SEED = 1
DISTRIBUTION_MODE = "easy"
NUM_ENVS = 256
ENV_ID = "chaser"
```
