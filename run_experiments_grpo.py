import subprocess
import sys
import time

def run_experiments():
    # 1. Define the variables to loop over
    envs = ["chaser", "fruitbot"]
    seed_values = [1, 2, 3]
    
    experiments = [

        ("GRPO Flavoured 2", "impoola_cnn.grpo_flavoured2_training")
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
        "--normalize-reward"  # Placeholder, will be replaced in the loop
    ]

    # 3. Nested Loop: Iterate through Environments, then Experiments
    for env in envs:
        for seed in seed_values:
            for exp_name, module_name in experiments:
            
                full_experiment_name = f"{exp_name} on {env}"
            
                print(f"\n{'='*60}")
                print(f"STARTING: {full_experiment_name}")
                print(f"{'='*60}\n")

                raw_command = [sys.executable, "-m", module_name, "--env-id", env, "--seed", seed] + common_args

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