# self_play.py

import os
import random
import time
import json
import uuid
import argparse
import wandb
import multiprocessing
from typing import List, Tuple
import torch
import numpy as np

# --- Local Project Imports ---
# Use the new generic game runner and utility functions
from alphazero.game_runner import GameConfig, play_game
from alphazero.utils import setup_selfplay_opening, format_time, manage_replay_buffer
from alphazero.env import ChessEnv
from alphazero.move_encoder import MoveEncoder

# --- Config Imports ---
from config import (
    # Project and File Paths
    PROJECT_NAME,
    TENSOR_CACHE_DIR,
    MODEL_DIR,
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
os.makedirs(TENSOR_CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

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

    game_examples, _, outcome_type, game_length = play_game(config, env)

    print(
        f"{prefix} Game finished. Outcome: {outcome_type}, Moves: {game_length}, Time: {format_time(time.time() - game_start_time)}",
        flush=True,
    )
    return game_examples, outcome_type, game_length


def process_results(results, args):
    """
    Processes the results from all self-play workers.
    Saves training data, aggregates stats, and logs them.
    """
    print(f"\nProcessing results from {len(results)} games...")

    summary_counts = {}
    game_lengths = []

    # This is a small object, creating it once is fine.
    encoder = MoveEncoder()

    for game_examples, outcome_type, game_length in results:
        summary_counts[outcome_type] = summary_counts.get(outcome_type, 0) + 1
        game_lengths.append(game_length)

        if game_examples:
            # We now process and save each position as a separate tensor file.
            for state_np, pi_np, z_value in game_examples:
                # --- Original Data Point ---
                state_tensor = torch.from_numpy(state_np.copy()).float()
                pi_tensor = torch.from_numpy(pi_np.copy()).float()
                z_tensor = torch.tensor([z_value], dtype=torch.float32)

                data_id = uuid.uuid4()
                torch.save(
                    (state_tensor, pi_tensor, z_tensor),
                    f"{TENSOR_CACHE_DIR}/data_{data_id}.pt",
                )

                # --- Augmented (Flipped) Data Point ---
                flipped_state_np = np.ascontiguousarray(np.flip(state_np, axis=2))

                original_pi_torch = torch.from_numpy(pi_np)
                flipped_pi_torch = torch.zeros_like(original_pi_torch)
                for i, prob in enumerate(original_pi_torch):
                    if prob > 0:
                        flipped_idx = encoder.flip_map[i]
                        flipped_pi_torch[flipped_idx] = prob

                flipped_state_tensor = torch.from_numpy(flipped_state_np).float()

                flipped_data_id = uuid.uuid4()
                torch.save(
                    (flipped_state_tensor, flipped_pi_torch, z_tensor),
                    f"{TENSOR_CACHE_DIR}/data_{flipped_data_id}.pt",
                )

    # --- Log summary statistics ---
    avg_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0

    selfplay_summary = {
        "selfplay_total_games": args.num_games,
        "selfplay_avg_game_length": avg_length,
    }
    selfplay_summary.update({f"outcome_{k}": v for k, v in summary_counts.items()})

    if wandb.run and not wandb.run.disabled:
        wandb.log(selfplay_summary)

    with open(args.result_file, "w") as f:
        json.dump(selfplay_summary, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
#  Main Execution Block
# ─────────────────────────────────────────────────────────────────────────────


def main(args):
    """Orchestrates the self-play data generation process using a pool of workers."""

    print(
        f"Starting self-play for {args.num_games} games using {NUM_SELFPLAY_WORKERS} parallel workers...\n"
    )

    wandb.init(
        project=PROJECT_NAME,
        group=args.gen_id,
        name=f"{args.gen_id}-self-play",
        job_type="self-play",
        mode="disabled" if args.no_wandb else "online",
    )

    start_time = time.time()

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=NUM_SELFPLAY_WORKERS) as pool:
        # map() blocks until all results are in. It distributes the work automatically.
        results = pool.map(run_selfplay_worker, range(args.num_games))

    print(
        f"\nAll self-play games finished in {format_time(time.time() - start_time)}. Processing results..."
    )

    process_results(results, args)

    manage_replay_buffer()

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run self-play games to generate training data."
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=DEFAULT_NUM_SELFPLAY_GAMES,
        help=f"Number of self-play games to generate. Defaults to {DEFAULT_NUM_SELFPLAY_GAMES}.",
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
