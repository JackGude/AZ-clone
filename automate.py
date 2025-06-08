# automate.py

import os
import sys
import subprocess
import shutil
import argparse
import wandb
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

# --- File Paths and Scripts ---
SELFPLAY_SCRIPT    = "self_play.py"
TRAIN_SCRIPT       = "train.py"
EVALUATE_SCRIPT    = "evaluate.py"

CHECKPOINT_DIR     = "checkpoints"
BEST_CHECKPOINT    = os.path.join(CHECKPOINT_DIR, "best.pth")
CANDIDATE_PATH     = os.path.join(CHECKPOINT_DIR, "candidate.pth")

# --- Pipeline Parameters ---
# You can tune these values to control the training process.
NUM_SELFPLAY_GAMES = 100    # Number of games to generate per generation
WIN_THRESHOLD      = 0.50   # Win rate needed for a candidate to be promoted (50% means not worse)
NUM_EVAL_GAMES     = 40     # Number of games to play to compare models
EVAL_TIME_LIMIT    = 5      # Time in seconds per move for evaluation games
WARMUP_GENS        = 1      # Number of initial generations to run without evaluation to bootstrap a model

# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline Steps
# ─────────────────────────────────────────────────────────────────────────────

def run_subprocess(cmd):
    """
    Runs a command as a subprocess, streams its output live, captures it for parsing,
    and raises an exception if the subprocess fails.
    """
    print(f"\n--- Running command: {' '.join(cmd)} ---")
    
    # Popen allows us to capture output in real-time.
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    
    output_lines = []
    # Read and print output line-by-line as it's generated
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        output_lines.append(line)
        
    process.wait() # Wait for the subprocess to finish
    
    if process.returncode != 0:
        raise RuntimeError(f"Execution of '{cmd[1]}' failed with exit code {process.returncode}")
        
    return "".join(output_lines)

def run_selfplay(generation):
    """Runs the self-play script to generate new training data."""
    print(f"\n=== [AUTO] Generation {generation:02d} --> Running Self-Play ===")
    cmd = [sys.executable, SELFPLAY_SCRIPT, "--num_games", str(NUM_SELFPLAY_GAMES)]
    run_subprocess(cmd)

def run_training(generation):
    """Runs the training script, which saves its best model as 'candidate.pth'."""
    print(f"\n=== [AUTO] Generation {generation:02d} --> Running Training ===")
    cmd = [sys.executable, TRAIN_SCRIPT] # Add training args here if needed
    run_subprocess(cmd)
    
    # Verify that the training process successfully created a candidate model
    if not os.path.exists(CANDIDATE_PATH):
        raise FileNotFoundError(f"Training completed, but the expected output '{CANDIDATE_PATH}' was not created.")

def run_evaluation(generation):
    """
    Runs the evaluation script and parses its output to get the win rate.
    """
    print(f"\n=== [AUTO] Generation {generation:02d} --> Running Evaluation ===")

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

# ─────────────────────────────────────────────────────────────────────────────
#  Main Automation Loop
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    """The main automation loop that drives the generations of self-play and training."""
    
    # Log in to Weights & Biases once at the start
    try:
        wandb.login()
    except Exception as e:
        print(f"Could not log into wandb automatically. Please run `wandb login` in your terminal. Error: {e}")
        return

    generation = args.start_generation
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # The main continuous loop
    while True:
        print(f"\n\n{'#'*20} Generation {generation:03d} {'#'*20}")

        # Start a new wandb run for this generation
        run_config = {
            "generation": generation,
            "num_selfplay_games": NUM_SELFPLAY_GAMES,
            "num_eval_games": NUM_EVAL_GAMES,
            "eval_time_limit_s": EVAL_TIME_LIMIT,
            "win_threshold": WIN_THRESHOLD,
        }
        run = wandb.init(
            project="alphazero-chess",
            name=f"generation-{generation:03d}",
            config=run_config
        )
        
        # This environment variable allows subprocesses to log to the same wandb run.
        os.environ["WANDB_RUN_ID"] = run.id

        # Step 1: Generate new self-play data using the current best model
        run_selfplay(generation)
        
        # Step 2: Train a new candidate model on the latest data
        run_training(generation)
        
        # Step 3: Evaluate the candidate against the current best
        is_warmup = (generation <= WARMUP_GENS)
        if is_warmup:
            print(f"\n--- Warm-up generation. Auto-promoting candidate. ---")
            was_promoted = promote_candidate(1.0) # Force promotion
        else:
            win_rate = run_evaluation(generation)
            was_promoted = promote_candidate(win_rate)

        # Log the final outcome of the generation to the dashboard
        run.log({"promoted_to_best": was_promoted, "eval_win_rate": win_rate if not is_warmup else 1.0})
        run.finish() # End the wandb run for this generation
        
        generation += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate the AlphaZero training loop with wandb.")
    parser.add_argument(
        "--start-generation",
        type=int,
        default=1,
        help="The generation number to start the loop from. Defaults to 1."
    )
    args = parser.parse_args()
    main(args)