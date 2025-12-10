import subprocess
import sys

# Hyperparameter values
rnd_coefs = [0.05, 0.5, 1]
rnd_lrs = [1e-4, 1e-3]
envs = ["chaser", "fruitbot"]

for env in envs:
    for coef in rnd_coefs:
        for lr in rnd_lrs:
            exp_name = f"ppo_rnd__coef{coef}_lr{lr}_{env}"

            cmd = [
                sys.executable,
                "-m",
                "impoola_cnn.ppo_training_rnd",
                "--rnd_coef",
                str(coef),
                "--rnd_lr",
                str(lr),
                "--rnd_output_size",
                str(128),
                "--exp_name",
                exp_name,
                "--output_dir",
                f"outputs_rnd_sweep/{exp_name}",
                "--env_id",
                env,
                "--distribution_mode",
                "easy",
            ]

            print("Running:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
