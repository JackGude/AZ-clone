# automate.py

import os
import sys
import subprocess
import shutil
import argparse
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

# --- File Paths and Scripts ---
SELFPLAY_SCRIPT    = "self_play.py"
TRAIN_SCRIPT       = "train.py"
EVALUATE_SCRIPT    = "evaluate.py"
STOP_FILE          = "stop.txt"     # If this file exists, the automation will stop after the current generation

CHECKPOINT_DIR     = "checkpoints"
BEST_CHECKPOINT    = os.path.join(CHECKPOINT_DIR, "best.pth")
CANDIDATE_PATH     = os.path.join(CHECKPOINT_DIR, "candidate.pth")

# --- Pipeline Parameters ---
# You can tune these values to control the training process.
NUM_SELFPLAY_GAMES = 100    # Number of games to generate per generation
WIN_THRESHOLD      = 0.50   # Win rate needed for a candidate to be promoted (50% means not worse)
NUM_EVAL_GAMES     = 12     # Number of games to play to compare models
EVAL_TIME_LIMIT    = 20      # Time in seconds per move for evaluation games
WARMUP_GENS        = 10      # Number of initial generations to run without evaluation to bootstrap a model

# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline Steps
# ─────────────────────────────────────────────────────────────────────────────

def run_subprocess(cmd):
    """
    Runs a command as a subprocess, streams its output live, captures it for parsing,
    and raises an exception if the subprocess fails.
    """
    print(f"\n{'='*20} Running command: {' '.join(cmd)} {'='*20}")
    
    # Popen allows us to capture output in real-time.
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        bufsize=1,  # Line buffered
        universal_newlines=True
    )
    
    output_lines = []
    # Read and print output line-by-line as it's generated
    for line in iter(process.stdout.readline, ''):
        print(line, end='', flush=True)  # Added flush=True for immediate output
        output_lines.append(line)
        
    process.wait() # Wait for the subprocess to finish
    
    if process.returncode != 0:
        raise RuntimeError(f"Execution of '{cmd[1]}' failed with exit code {process.returncode}")
    
    print(f"\n{'='*20} Command completed {'='*20}\n")
    return "".join(output_lines)

def run_selfplay(generation_id_str):
    """Runs the self-play script to generate new training data."""
    print(f"\n=== [AUTO] {generation_id_str} --> Running Self-Play ===")
    cmd = [
        sys.executable, 
        SELFPLAY_SCRIPT, 
        "--num_games", str(NUM_SELFPLAY_GAMES),
        "--gen-id", generation_id_str
    ]
    run_subprocess(cmd)

def run_training(generation_id_str):
    """Runs the training script, which saves its best model as 'candidate.pth'."""
    print(f"\n=== [AUTO] {generation_id_str} --> Running Training ===")
    cmd = [
        sys.executable, 
        TRAIN_SCRIPT,
        "--gen-id", generation_id_str
    ]
    run_subprocess(cmd)
    
    # Verify that the training process successfully created a candidate model
    if not os.path.exists(CANDIDATE_PATH):
        raise FileNotFoundError(f"Training completed, but the expected output '{CANDIDATE_PATH}' was not created.")

def run_evaluation(generation_id_str):
    """
    Runs the evaluation script and parses its output to get the win rate.
    """
    print(f"\n=== [AUTO] {generation_id_str} --> Running Evaluation ===")

    # If no 'best' model exists yet (first run), the candidate wins by default.
    if not os.path.exists(BEST_CHECKPOINT):
        print("No existing 'best.pth' found. The candidate will be automatically promoted.")
        return 1.0

    cmd = [
        sys.executable, EVALUATE_SCRIPT,
        "--old", BEST_CHECKPOINT,
        "--new", CANDIDATE_PATH,
        "--games", str(NUM_EVAL_GAMES),
        "--time-limit", str(EVAL_TIME_LIMIT),
        "--gen-id", generation_id_str
    ]
    
    # Capture the full output of the script to find the result line.
    output = run_subprocess(cmd)
    
    # Parse the win rate from the last line of the script's output for robustness.
    for line in reversed(output.strip().split('\n')):
        if line.startswith("FINAL_WIN_RATE:"):
            win_rate = float(line.split(":")[1].strip())
            print(f"--- Parsed win rate from evaluation: {win_rate:.3f} ---")
            return win_rate
    
    raise RuntimeError("Could not parse win rate from evaluate.py output. Check for errors in the script.")

def promote_candidate(win_rate):
    """
    If the candidate's win rate is above the threshold, it replaces 'best.pth'.
    """
    if win_rate >= WIN_THRESHOLD:
        print(f"→ Candidate met win threshold ({win_rate:.2f} >= {WIN_THRESHOLD}). Promoting to 'best.pth'.")
        # Use rename for an atomic operation, replacing the old best model.
        shutil.move(CANDIDATE_PATH, BEST_CHECKPOINT)
        return True
    else:
        print(f"→ Candidate failed to meet win threshold ({win_rate:.2f} < {WIN_THRESHOLD}). Discarding.")
        os.remove(CANDIDATE_PATH)
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
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)  # Clean up the file for the next time

    generation = args.start_generation

    print(f"\n{'#'*20} Starting AlphaZero Training Loop {'#'*20}")
    print(f"Starting from generation {generation}")
    print(f"Configuration:")
    print(f"  - Self-play games per generation: {NUM_SELFPLAY_GAMES}")
    print(f"  - Evaluation games: {NUM_EVAL_GAMES}")
    print(f"  - Win threshold: {WIN_THRESHOLD}")
    print(f"  - Warm-up generations: {WARMUP_GENS}")
    print(f"{'#'*80}\n")

    # The main continuous loop
    while True:
        generation_id_str = f"gen-{generation:03d}"
        print(f"\n{'#'*20} {generation_id_str} {'#'*20}")

        # Step 1: Generate new self-play data using the current best model
        print(f"\n>>> Step 1: Running Self-Play for {generation_id_str}")
        run_selfplay(generation_id_str)
        
        # Step 2: Train a new candidate model on the latest data
        print(f"\n>>> Step 2: Running Training for {generation_id_str}")
        run_training(generation_id_str)
        
        # Step 3: Evaluate the candidate against the current best
        print(f"\n>>> Step 3: Running Evaluation for {generation_id_str}")
        is_warmup = (generation <= WARMUP_GENS)
        if is_warmup:
            print(f"\n--- Warm-up generation. Auto-promoting candidate. ---")
            promote_candidate(1.0) # Force promotion
        else:
            win_rate = run_evaluation(generation_id_str)
            promote_candidate(win_rate)
        
        print(f"\n{'#'*20} Completed {generation_id_str} {'#'*20}\n")

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