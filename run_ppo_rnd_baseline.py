import subprocess
import sys

envs = ["chaser", "fruitbot"]
seeds = [1, 2, 3]

for env in envs:
    for seed in seeds:
        exp_name = f"ppo_rnd_baseline_seed_{seed}_{env}"
        cmd = [
            sys.executable, "-m", "impoola_cnn.ppo_training_rnd",
            "--exp_name", exp_name,
            "--seed", str(seed),
            "--env_id", env,
            "--output_dir", f"ppo_rnd_baseline_3x/{exp_name}",
            "--n_datapoints_csv", "1000"
        ]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd)
