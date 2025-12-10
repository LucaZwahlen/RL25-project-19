import subprocess
import sys

envs = ["chaser", "fruitbot"]
seeds = [1, 2, 3]

common_args = [
    "--num-envs", "90",
    "--num-steps", "64",
    "--latent-space-dim", "256",
    "--cnn-filters", "16", "32", "32",
    "--learning-rate", "0.0006",
    "--update-epochs", "1",
    "--num-minibatches", "8",
    "--ent-coef", "0.01",
    "--total-timesteps", "25000000",  # Set small for testing, increase for real run
    "--normalize-reward"  # Placeholder, will be replaced in the loop
]

for env in envs:
    for seed in seeds:
        exp_name = f"grpo_baseline_seed_{seed}_{env}"
        cmd = [
            sys.executable, "-m", "impoola_cnn.grpo_flavoured2_training",
            "--exp_name", exp_name,
            "--seed", str(seed),
            "--env_id", env,
            "--output_dir", f"grpo_baseline/{exp_name}",
            "--n_datapoints_csv", "500"
        ] + common_args
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd)
