# self_play.py

import os
import random
import pickle
import time
import json
import uuid
import argparse
import wandb
import multiprocessing
from typing import List, Tuple

# --- Local Project Imports ---
# Use the new generic game runner and utility functions
from alphazero.game_runner import GameConfig, play_game
from alphazero.utils import setup_selfplay_opening, format_time, manage_replay_buffer
from alphazero.env import ChessEnv

# --- Config Imports ---
from config import (
    # Project and File Paths
    PROJECT_NAME,
    REPLAY_DIR,
    CHECKPOINT_DIR,
    BEST_MODEL_PATH,
    # Self-Play Config
    DEFAULT_NUM_SELFPLAY_GAMES,
    NUM_SELFPLAY_WORKERS,
    SELFPLAY_CPUCT,
    DIRICHLET_EPSILON,
    DIRICHLET_ALPHA,
    TEMP_THRESHOLD,
    SELFPLAY_MAX_MOVES,
    RESIGN_THRESHOLD,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Ensure necessary directories exist, paths come from config.py
os.makedirs(REPLAY_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


def choose_time_limit(prefix: str):
    """Randomly chooses a time limit to add variety to the training data."""
    if random.random() < 0.5:
        time_limit = random.uniform(0.5, 1.5)
    else:
        time_limit = random.uniform(2.0, 4.0)
    print(f"{prefix} Time limit per move: {time_limit:.1f}s", flush=True)
    return time_limit


def run_selfplay_worker(game_num: int) -> Tuple[List[dict], str, int]:
    """
    This is the target function for each worker process. It plays one game.
    """
    prefix = f"[Worker {game_num + 1}]"
    print(f"{prefix} Starting game...", flush=True)
    game_start_time = time.time()

    # Create a configuration for a self-play game
    config = GameConfig(
        white_model_path=BEST_MODEL_PATH,
        black_model_path=BEST_MODEL_PATH,
        time_limit=choose_time_limit(prefix),
        c_puct=SELFPLAY_CPUCT,
        use_dirichlet_noise=True,
        dirichlet_alpha=DIRICHLET_ALPHA,
        dirichlet_epsilon=DIRICHLET_EPSILON,
        use_temperature_sampling=True,
        temp_threshold=TEMP_THRESHOLD,
        use_adjudication=False,  # Adjudication is typically not used for self-play
        win_adjudication_threshold=0.99,  # N/A
        win_adjudication_patience=1,  # N/A
        draw_adjudication_threshold=0.0,  # N/A
        draw_adjudication_patience=1,  # N/A
        resign_threshold=RESIGN_THRESHOLD,
        selfplay_max_moves=SELFPLAY_MAX_MOVES,
        verbose=False,
        log_prefix=prefix,
    )

    env = ChessEnv()
    # Pass the game_num to the setup function for clearer logging
    setup_selfplay_opening(env, game_num)

    game_examples, outcome_type, game_length = play_game(config, env)

    print(
        f"{prefix} Game finished. Outcome: {outcome_type}, Moves: {game_length}, Time: {format_time(time.time() - game_start_time)}",
        flush=True,
    )
    return game_examples, outcome_type, game_length


# ─────────────────────────────────────────────────────────────────────────────
#  Main Execution Block
# ─────────────────────────────────────────────────────────────────────────────


def main(args):
    """Orchestrates the self-play data generation process using a pool of workers."""
    # Initialize wandb for the main process
    wandb.init(
        project=PROJECT_NAME,
        group=args.gen_id,
        name=f"{args.gen_id}-self-play",
        job_type="self-play",
        mode="disabled" if args.no_wandb else "online",
    )

    print(
        f"Starting self-play for {args.num_games} games using {NUM_SELFPLAY_WORKERS} parallel workers..."
    )
    start_time = time.time()

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=NUM_SELFPLAY_WORKERS) as pool:
        # map() blocks until all results are in. It distributes the work automatically.
        results = pool.map(run_selfplay_worker, range(args.num_games))

    print(
        f"\nAll self-play games finished in {format_time(time.time() - start_time)}. Processing results..."
    )

    # Process the results from all workers
    summary_counts = {}
    game_lengths = []

    for game_examples, outcome_type, game_length in results:
        summary_counts[outcome_type] = summary_counts.get(outcome_type, 0) + 1
        game_lengths.append(game_length)

        if game_examples:
            game_id = uuid.uuid4()
            game_path = os.path.join(REPLAY_DIR, f"game_{outcome_type}_{game_id}.pkl")
            with open(game_path, "wb") as f:
                pickle.dump(game_examples, f)

    manage_replay_buffer()

    # Log summary statistics
    avg_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0

    selfplay_summary = {
        "selfplay_total_games": args.num_games,
        "selfplay_avg_game_length": avg_length,
    }
    selfplay_summary.update({f"outcome_{k}": v for k, v in summary_counts.items()})

    wandb.log(selfplay_summary)

    with open(args.result_file, "w") as f:
        json.dump(selfplay_summary, f, indent=2)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run self-play games to generate training data."
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=DEFAULT_NUM_SELFPLAY_GAMES,
        help="Number of self-play games to generate.",
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable wandb logging for this run."
    )
    parser.add_argument(
        "--gen-id", type=str, default="manual", help="Generation ID for this run."
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default="logs/selfplay_results.json",
        help="Path to write the self-play results JSON file.",
    )
    args = parser.parse_args()
    main(args)
