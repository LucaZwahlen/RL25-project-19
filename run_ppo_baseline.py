import subprocess
import sys

envs = ["fruitbot", "chaser"]
seeds = [2, 3, 1]

for s in seeds:
    for env in envs:
        exp_name = f"ppo_baseline_seed_{s}_{env}"
        cmd = [
            sys.executable, "-m", "impoola_cnn.ppo_training",
            "--seed", str(s),
            "--exp_name", exp_name,
            "--env_id", env,
            "--encoder_type", "impala",
            "--output_dir", f"ppo_baseline_3x/{exp_name}",
        ]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd)
