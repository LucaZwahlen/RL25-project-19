import subprocess
import sys

envs = ["fruitbot", "chaser"]
seeds = [1]

for s in seeds:
    for env in envs:
        exp_name = f"dqn_baseline_seed_{s}_{env}"
        cmd = [
            sys.executable, "-m", "impoola_cnn.dqn_training",
            "--seed", str(s),
            "--exp_name", exp_name,
            "--env_id", env,
            "--encoder_type", "impala",
            "--output_dir", f"dqn_baseline/{exp_name}",
        ]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd)
