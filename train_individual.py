import subprocess
import sys
import os

def run_experiments():
    """
    Orchestrates the training jobs sequentially to prevent GPU OOM errors.
    """
    
    # List of experiments to run
    # Each dictionary contains the parameters for a specific training run
    experiments = [
        {"dataset": "eth", "tag": "social_implicit_eth_test", "w_norm": 0.00001, "w_cos": 0.0001},
        {"dataset": "hotel", "tag": "social_implicit_hotel_test", "w_norm": 0.00001, "w_cos": 0.00001},
        {"dataset": "univ", "tag": "social_implicit_univ_test", "w_norm": 0.00001, "w_cos": 0.0001},
        {"dataset": "zara1", "tag": "social_implicit_zara1_test", "w_norm": 0.00001, "w_cos": 0.0001},
        {"dataset": "zara1", "tag": "social_implicit_zara2_test", "w_norm": 0.0001, "w_cos": 0.0001},
        {"dataset": "sdd", "tag": "social_implicit_sdd_test", "w_norm": 0.0001, "w_cos": 0.0001},
    ]

    print(f"Starting sequential training for {len(experiments)} experiments...")

    for i, exp in enumerate(experiments):
        print(f"\n{'='*20}")
        print(f"Launching Experiment {i+1}/{len(experiments)}: {exp['dataset']}")
        print(f"{'='*20}\n")

        # Prepare the command list
        # sys.executable ensures we use the exact same python interpreter as this script
        cmd = [
            sys.executable, "train.py",
            "--lr", "1.0",
            "--dataset", exp["dataset"],
            "--tag", exp["tag"],
            "--w_norm", str(exp["w_norm"]),
            "--w_cos", str(exp["w_cos"])
        ]

        try:
            # subprocess.run blocks execution until the training finishes
            # This ensures they run one by one and not in parallel
            result = subprocess.run(cmd, check=True)
            print(f"Successfully finished: {exp['dataset']}")
        
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running {exp['dataset']}: {e}")
            continue # Continue to the next experiment even if this one failed

    print("\nAll experiments have been processed.")

if __name__ == '__main__':
    run_experiments()