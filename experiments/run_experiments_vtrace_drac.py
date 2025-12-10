import subprocess
import sys

envs = ["chaser", "fruitbot"]
drac = [0.02, 0.04, 0.06, 0.08, 0.1]
seeds = [0, 1]

for env in envs:
    for s in seeds:
        for d in drac:
            exp_name = f"vtrace_drac_final_{d}_{env}_seed{s}"
            cmd = [
                sys.executable, "-m", "impoola_cnn.vtrace_training",
                "--drac_lambda", str(d),
                "--exp_name", exp_name,
                "--env_id", env,
                "--encoder_type", "impala",
                "--seed", str(s),
                "--output_dir", f"vtrace_drac/{exp_name}",
            ]
            print("Running:", " ".join(cmd), flush=True)
            subprocess.run(cmd)
