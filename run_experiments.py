import subprocess
import sys


envs = ["chaser", "fruitbot"]
drac = [0.05, 0.0]
ent_coef = [0.011, 0.1, 0.009]

for d in drac:
    for c in ent_coef:
        for env in envs:
            exp_name = f"vtrace_drac_{d}_entr_{c}_{env}"
            cmd = [
                sys.executable, "-m", "impoola_cnn.vtrace_training",
                "--drac_lambda", str(d),
                "--ent_coef", str(c),
                "--exp_name", exp_name,
                "--env_id", env,
                "--encoder_type", "impala",
                "--output_dir", f"vtrace_drac_entropy/{exp_name}",
            ]
            print("Running:", " ".join(cmd), flush=True)
            subprocess.run(cmd)