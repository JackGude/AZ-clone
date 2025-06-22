# automate.py

import os
import sys
import time
import signal
import subprocess
import argparse
import shutil
from config import (
    # Project and File Paths
    MODEL_DIR,
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
    AUTOMATE_EVAL_TIME_LIMIT,
)
import json
from alphazero.utils import format_time, ensure_project_root


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

current_subprocess = None

# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline Steps
# ─────────────────────────────────────────────────────────────────────────────


def run_subprocess(cmd, log_path):
    """
    Runs a command as a subprocess, allowing the main script to remain responsive to signals.
    """
    global current_subprocess

    # Create directories for the log and results
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    env = os.environ.copy()
    project_root = os.path.dirname(os.path.abspath(__file__))
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

    with open(log_path, "w") as log_file:
        # Use Popen to run the command with the correct environment
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            ),
            preexec_fn=(os.setsid if sys.platform != "win32" else None),
        )
        current_subprocess = process
        return_code = process.wait()

    process_was_terminated_by_handler = current_subprocess is None and return_code != 0
    current_subprocess = None

    if return_code != 0 and not process_was_terminated_by_handler:
        error_message = (
            f"Execution of '{' '.join(cmd)}' failed with exit code {return_code}.\n"
            f"Check the log file for details: {log_path}"
        )
        raise RuntimeError(error_message)


def run_selfplay(generation_id_str):
    """Runs the self-play script to generate new training data."""
    print(f"\n=== [AUTO] {generation_id_str} --> Running Self-Play ===")
    log_path = os.path.join(LOGS_DIR, f"{generation_id_str}/selfplay.log")

    cmd = [
        sys.executable,
        SELFPLAY_SCRIPT,
        "--num-games",
        str(AUTOMATE_NUM_SELFPLAY_GAMES),
        "--gen-id",
        generation_id_str
    ]
    run_subprocess(cmd, log_path)


def run_training(generation_id_str, is_warmup):
    """Runs the training script, which saves its best model as 'candidate.pth'."""
    print(f"\n=== [AUTO] {generation_id_str} --> Running Training ===")
    log_path = os.path.join(LOGS_DIR, f"{generation_id_str}/training.log")

    cmd = [
        sys.executable,
        TRAIN_SCRIPT,
        "--gen-id",
        generation_id_str,
    ]
    
    if not is_warmup:
        cmd.append("--load-weights")

    run_subprocess(cmd, log_path)

    # Verify that the training process successfully created a candidate model
    if not os.path.exists(CANDIDATE_MODEL_PATH):
        raise FileNotFoundError(
            f"Training completed, but the expected output '{CANDIDATE_MODEL_PATH}' was not created."
        )


def run_evaluation(generation_id_str, is_warmup=False):
    """
    Runs the evaluation script and promotes the candidate if it meets the win threshold.
    If is_warmup is True, automatically promotes the candidate without evaluation.
    """
    # If no 'best' model exists yet (first run), the candidate wins by default.
    if not os.path.exists(BEST_MODEL_PATH):
        print("\n--- No previous best model found. Promoting candidate. ---")
        promote_candidate(1.0)
        return

    if is_warmup:
        print("\n--- Warm-up generation. Auto-promoting candidate. ---")
        promote_candidate(1.0)
        return

    log_path = os.path.join(LOGS_DIR, f"{generation_id_str}/evaluation.log")
    result_path = os.path.join(LOGS_DIR, f"{generation_id_str}/eval_result.json")

    cmd = [
        sys.executable,
        EVALUATE_SCRIPT,
        "--games",
        str(AUTOMATE_NUM_EVAL_GAMES),
        "--time-limit",
        str(AUTOMATE_EVAL_TIME_LIMIT),
        "--gen-id",
        generation_id_str,
        "--result-file",
        result_path,
    ]

    run_subprocess(cmd, log_path)

    try:
        with open(result_path, "r") as f:
            result = json.load(f)
            win_rate = float(result["win_rate"])

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
        print(
            f"\n→ Candidate met win threshold ({win_rate:.2f} >= {AUTOMATE_WIN_THRESHOLD}). Promoting to 'best.pth'."
        )
        # Use rename for an atomic operation, replacing the old best model.
        shutil.move(CANDIDATE_MODEL_PATH, BEST_MODEL_PATH)
        return True
    else:
        print(
            f"\n→ Candidate failed to meet win threshold ({win_rate:.2f} < {AUTOMATE_WIN_THRESHOLD}). Discarding."
        )
        os.remove(CANDIDATE_MODEL_PATH)
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Main Automation Loop
# ─────────────────────────────────────────────────────────────────────────────


def signal_handler(sig, frame):
    """
    Handles CTRL+C by forcefully terminating the current subprocess and its entire process tree.
    """
    global current_subprocess
    print("\n\nCTRL+C detected! Shutting down...")

    if current_subprocess:
        print(f"--> Terminating subprocess tree for PID {current_subprocess.pid}")
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(current_subprocess.pid)],
                    check=True,
                    capture_output=True,
                )
            else:
                os.killpg(os.getpgid(current_subprocess.pid), signal.SIGKILL)
            print("--> Subprocess tree terminated successfully.")
        except Exception as e:
            print(f"--> Note: Subprocess may have already terminated. {e}")

    sys.exit(0)


def main(args):
    """The main automation loop that drives the generations of self-play and training."""

    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)

    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)

    generation = args.start_generation
    current_step = args.start_step

    print(f"\n{'#' * 20} Starting AlphaZero Training Loop {'#' * 20}")
    print(f"Starting from generation {generation}")
    print("Configuration:")
    print(f"  - Self-play games per generation: {AUTOMATE_NUM_SELFPLAY_GAMES}")
    print(f"  - Evaluation games: {AUTOMATE_NUM_EVAL_GAMES}")
    print(f"  - Win threshold: {AUTOMATE_WIN_THRESHOLD}")
    print(f"  - Warm-up generations: {AUTOMATE_WARMUP_GENS}")
    print(f"{'#' * 80}\n")

    while True:
        is_warmup = generation <= AUTOMATE_WARMUP_GENS
        generation_id_str = f"gen-{generation:03d}"
        print(f"\n{'#' * 20} {generation_id_str} {'#' * 20}")

        # Step 1: Generate new self-play data
        if current_step == "selfplay":
            print(f"\n>>> Step 1: Running Self-Play for {generation_id_str}")
            selfplay_start_time = time.time()
            run_selfplay(generation_id_str)
            print(
                f"\n<<< Self-Play finished in {format_time(time.time() - selfplay_start_time)}"
            )
            current_step = "training"

        # Step 2: Train a new candidate model
        if current_step == "training":
            print(f"\n>>> Step 2: Running Training for {generation_id_str}")
            training_start_time = time.time()
            run_training(generation_id_str, is_warmup)
            print(
                f"\n<<< Training finished in {format_time(time.time() - training_start_time)}"
            )
            current_step = "evaluation"

        # Step 3: Evaluate the candidate
        if current_step == "evaluation":
            print(f"\n>>> Step 3: Running Evaluation for {generation_id_str}")
            evaluation_start_time = time.time()
            run_evaluation(generation_id_str, is_warmup)
            print(
                f"\n<<< Evaluation finished in {format_time(time.time() - evaluation_start_time)}"
            )
            current_step = "selfplay"

        if os.path.exists(STOP_FILE):
            print(
                "\n[AUTO] 'stop.txt' file detected. Shutting down gracefully after this generation.",
                flush=True,
            )
            os.remove(STOP_FILE)
            break

        generation += 1


if __name__ == "__main__":
    ensure_project_root()
    parser = argparse.ArgumentParser(
        description="Automate the AlphaZero training loop."
    )
    parser.add_argument(
        "--start-generation",
        type=int,
        default=1,
        help="The generation number to start the loop from. Defaults to 1.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=100,
        help="The number of generations to run. Defaults to 100.",
    )
    parser.add_argument(
        "--start-step",
        type=str,
        default="selfplay",
        help="The step to start the loop from. Defaults to selfplay.",
        choices=["selfplay", "training", "evaluation"],
    )
    args = parser.parse_args()
    main(args)
