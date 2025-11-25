import subprocess
import sys

envs = ["chaser", "fruitbot"]
seeds = [1, 2, 3]

for env in envs:
    for seed in seeds:
        exp_name = f"vtrace_baseline_seed_{seed}_{env}"
        cmd = [
            sys.executable, "-m", "impoola_cnn.vtrace_training",
            "--exp_name", exp_name,
            "--seed", str(seed),
            "--env_id", env,
            "--output_dir", f"vtrace_baseline_3x/{exp_name}",
            "--n_datapoints_csv", "1000"
        ]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd)
