import subprocess
import sys
import time

def run_experiments():
    # 1. Define the variables to loop over
    envs = ["chaser", "fruitbot"]
    
    experiments = [
        ("Normal GRPO", "impoola_cnn.grpo_training"),
        ("GRPO Flavoured 1", "impoola_cnn.grpo_flavoured1_training"),
        ("GRPO Flavoured 2", "impoola_cnn.grpo_flavoured2_training"),
    ]

    # 2. Define arguments shared across all experiments
    # NOTE: We removed "--env-id" from here. We will add it dynamically in the loop.
    common_args = [
        "--num-envs", "90",
        "--num-steps", "64",
        "--latent-space-dim", "256",
        "--cnn-filters", "16", "32", "32",
        "--learning-rate", "0.0006",
        "--update-epochs", "1",
        "--num-minibatches", "8",
        "--ent-coef", "0.01",
        "--total-timesteps", "25000000", # Set small for testing, increase for real run
        "--normalize-reward",
        "--seed", "1"
    ]

    # 3. Nested Loop: Iterate through Environments, then Experiments
    for env in envs:
        for exp_name, module_name in experiments:
            
            full_experiment_name = f"{exp_name} on {env}"
            
            print(f"\n{'='*60}")
            print(f"STARTING: {full_experiment_name}")
            print(f"{'='*60}\n")

            # Construct the command:
            # python -m <module> --env-id <current_env> <common_args>
            raw_command = [sys.executable, "-m", module_name, "--env-id", env] + common_args

            # --- SANITIZATION (Safety Check) ---
            # This ensures no nested lists or integers break the subprocess call
            final_command = []
            for arg in raw_command:
                if isinstance(arg, list):
                    final_command.extend([str(item) for item in arg])
                else:
                    final_command.append(str(arg))
            # -----------------------------------

            start_time = time.time()
            
            try:
                # Run the command
                subprocess.run(final_command, check=True)
                
            except subprocess.CalledProcessError as e:
                print(f"\n[ERROR] {full_experiment_name} failed with error code {e.returncode}.")
                # Optional: break # Uncomment to stop everything on first error
            except KeyboardInterrupt:
                print("\n[STOPPED] Execution interrupted by user.")
                sys.exit(1)
                
            duration = (time.time() - start_time) / 60
            print(f"\n>>> Finished {full_experiment_name} in {duration:.2f} minutes.")

if __name__ == "__main__":
    run_experiments()