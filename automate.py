# automate.py

import os
import sys
import time
import subprocess
import shutil
import argparse
from datetime import datetime
from config import (
    # Project and File Paths
    CHECKPOINT_DIR,
    LOGS_DIR,
    BEST_MODEL_PATH,
    CANDIDATE_MODEL_PATH,
    SELFPLAY_SCRIPT,
    TRAIN_SCRIPT,
    EVALUATE_SCRIPT,
    STOP_FILE,
    # Automation Pipeline Config
    AUTOMATE_NUM_SELFPLAY_GAMES,
    AUTOMATE_NUM_EVAL_GAMES,
    AUTOMATE_WIN_THRESHOLD,
    AUTOMATE_WARMUP_GENS,
    AUTOMATE_EVAL_TIME_LIMIT    
)
import json

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

# The directory to store checkpoints.
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline Steps
# ─────────────────────────────────────────────────────────────────────────────

def run_subprocess(cmd, log_path):
    """
    Runs a command as a subprocess and redirects all output to a log file.
    
    Args:
        cmd: List of command and arguments to run
        log_path: Path to the log file where output will be written
    
    Raises:
        RuntimeError: If the subprocess fails
    """
    # Ensure the log directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    with open(log_path, 'w') as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True
        )
        process.wait()
    
    if process.returncode != 0:
        raise RuntimeError(f"Execution of '{cmd[1]}' failed with exit code {process.returncode}")

def run_selfplay(generation_id_str):
    """Runs the self-play script to generate new training data."""
    print(f"\n=== [AUTO] {generation_id_str} --> Running Self-Play ===")
    log_path = os.path.join(LOGS_DIR, f"{generation_id_str}_selfplay.log")
    result_path = os.path.join(LOGS_DIR, f"{generation_id_str}_selfplay_result.json")
    
    cmd = [
        sys.executable, 
        SELFPLAY_SCRIPT, 
        "--num_games", str(AUTOMATE_NUM_SELFPLAY_GAMES),
        "--gen-id", generation_id_str,
        "--result-file", result_path
    ]
    run_subprocess(cmd, log_path)

def run_training(generation_id_str, is_warmup):
    """Runs the training script, which saves its best model as 'candidate.pth'."""
    print(f"\n=== [AUTO] {generation_id_str} --> Running Training ===")
    log_path = os.path.join(LOGS_DIR, f"{generation_id_str}_training.log")
    result_path = os.path.join(LOGS_DIR, f"{generation_id_str}_training_result.json")
    
    cmd = [
        sys.executable, 
        TRAIN_SCRIPT,
        "--gen-id", generation_id_str,
        "--result-file", result_path,
        "--load-weights" if not is_warmup else ""
    ]
    run_subprocess(cmd, log_path)
    
    # Verify that the training process successfully created a candidate model
    if not os.path.exists(CANDIDATE_MODEL_PATH):
        raise FileNotFoundError(f"Training completed, but the expected output '{CANDIDATE_MODEL_PATH}' was not created.")

def run_evaluation(generation_id_str, is_warmup=False):
    """
    Runs the evaluation script and promotes the candidate if it meets the win threshold.
    If is_warmup is True, automatically promotes the candidate without evaluation.
    """
    # If no 'best' model exists yet (first run), the candidate wins by default.
    if not os.path.exists(BEST_MODEL_PATH):
        promote_candidate(1.0)
        return

    if is_warmup:
        print(f"\n--- Warm-up generation. Auto-promoting candidate. ---")
        promote_candidate(1.0)
        return

    log_path = os.path.join(LOGS_DIR, f"{generation_id_str}_evaluation.log")
    result_path = os.path.join(LOGS_DIR, f"{generation_id_str}_eval_result.json")
    
    cmd = [
        sys.executable, EVALUATE_SCRIPT,
        "--games", str(AUTOMATE_NUM_EVAL_GAMES),
        "--time-limit", str(AUTOMATE_EVAL_TIME_LIMIT),
        "--gen-id", generation_id_str,
        "--result-file", result_path
    ]
    
    run_subprocess(cmd, log_path)
    
    try:
        with open(result_path, 'r') as f:
            result = json.load(f)
            win_rate = float(result['win_rate'])
            
        # Clean up the result file
        os.remove(result_path)
        promote_candidate(win_rate)
        
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise RuntimeError(f"Failed to parse evaluation result file: {e}")
    except FileNotFoundError:
        raise RuntimeError("Evaluation completed but result file was not created")

def promote_candidate(win_rate):
    """
    If the candidate's win rate is above the threshold, it replaces 'best.pth'.
    """
    if win_rate >= AUTOMATE_WIN_THRESHOLD:
        print(f"→ Candidate met win threshold ({win_rate:.2f} >= {AUTOMATE_WIN_THRESHOLD}). Promoting to 'best.pth'.")
        # Use rename for an atomic operation, replacing the old best model.
        shutil.move(CANDIDATE_MODEL_PATH, BEST_MODEL_PATH)
        return True
    else:
        print(f"→ Candidate failed to meet win threshold ({win_rate:.2f} < {AUTOMATE_WIN_THRESHOLD}). Discarding.")
        os.remove(CANDIDATE_MODEL_PATH)
        return False
    
def format_time(seconds):
    """Formats a time in seconds into a human-readable string."""
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# ─────────────────────────────────────────────────────────────────────────────
#  Main Automation Loop
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    """The main automation loop that drives the generations of self-play and training."""
    
    # This check ensures that if a 'stop.txt' was left over from a previous
    # run, it won't prevent the script from starting.
    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)  # Clean up the file for the next time

    generation = args.start_generation

    print(f"\n{'#'*20} Starting AlphaZero Training Loop {'#'*20}")
    print(f"Starting from generation {generation}")
    print(f"Configuration:")
    print(f"  - Self-play games per generation: {AUTOMATE_NUM_SELFPLAY_GAMES}")
    print(f"  - Evaluation games: {AUTOMATE_NUM_EVAL_GAMES}")
    print(f"  - Win threshold: {AUTOMATE_WIN_THRESHOLD}")
    print(f"  - Warm-up generations: {AUTOMATE_WARMUP_GENS}")
    print(f"{'#'*80}\n")

    # The main continuous loop
    while True:
        is_warmup = (generation <= AUTOMATE_WARMUP_GENS)
        generation_id_str = f"gen-{generation:03d}"
        print(f"\n{'#'*20} {generation_id_str} {'#'*20}")

        # Step 1: Generate new self-play data
        print(f"\n>>> Step 1: Running Self-Play for {generation_id_str}")
        selfplay_start_time = time.time()
        run_selfplay(generation_id_str)
        print(f"<<< Self-Play finished in {format_time(time.time() - selfplay_start_time)}")
        
        # Step 2: Train a new candidate model
        print(f"\n>>> Step 2: Running Training for {generation_id_str}")
        training_start_time = time.time()
        run_training(generation_id_str, is_warmup)
        print(f"<<< Training finished in {format_time(time.time() - training_start_time)}")

        # Step 3: Evaluate the candidate
        print(f"\n>>> Step 3: Running Evaluation for {generation_id_str}")
        evaluation_start_time = time.time()
        run_evaluation(generation_id_str, is_warmup)
        print(f"<<< Evaluation finished in {format_time(time.time() - evaluation_start_time)}")

        if os.path.exists(STOP_FILE):
            print("\n[AUTO] 'stop.txt' file detected. Shutting down gracefully after this generation.", flush=True)
            os.remove(STOP_FILE)  # Clean up the file for the next time
            break # Exit the while loop

        generation += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate the AlphaZero training loop.")
    parser.add_argument(
        "--start-generation",
        type=int,
        default=1,
        help="The generation number to start the loop from. Defaults to 1."
    )
    args = parser.parse_args()
    main(args)