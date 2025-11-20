import subprocess
import sys


envs = ["chaser", "fruitbot"]
drac = [0.01, 0.1]

for env in envs:
    for d in drac:
        exp_name = f"vtrace_drac_new_{d}_{env}"
        cmd = [
            sys.executable, "-m", "impoola_cnn.vtrace_training",
            "--drac_lambda", str(d),
            # "--ent_coef", str(c),
            "--exp_name", exp_name,
            "--env_id", env,
            "--encoder_type", "impala",
            "--output_dir", f"vtrace_drac_entropy/{exp_name}",
        ]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd)