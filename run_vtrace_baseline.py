import subprocess
import sys

envs = ["chaser", "fruitbot"]
seeds = [1, 2, 3]

for env in envs:
    for seed in seeds:
        exp_name = f"vtrace_baseline_{env}"
        cmd = [
            sys.executable, "-m", "impoola_cnn.vtrace_training",
            "--exp_name", exp_name,
            "--seed", str(seed),
            "--env_id", env,
            "--output_dir", f"vtrace_baseline/{exp_name}",
        ]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd)
