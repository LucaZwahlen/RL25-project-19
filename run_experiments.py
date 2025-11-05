import subprocess
import sys

p_augments = [0.00, 0.05, 0.10]
micro_dropouts = [0.0, 0.005, 0.01, 0.1]

for p in p_augments:
    for m in micro_dropouts:
        cmd = [
            sys.executable, "-m", "impoola_cnn.vtrace_training",
            "--p_augment", str(p),
            "--micro_dropout_p", str(m),
            "--exp_name", f"overnight_vtrace__p{p}_m{m}",
            "--env_id", "chaser",
            "--encoder_type", "impala_new",
        ]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
