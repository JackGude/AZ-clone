# play.py

import os
import random
import time
import uuid
import argparse
import wandb
import multiprocessing
from typing import List, Tuple
import torch
import numpy as np
import pandas as pd

# --- Local Project Imports ---
from alphazero.game_runner import GameConfig, play_game
from alphazero.utils import format_time, ensure_project_root
from alphazero.env import ChessEnv
from alphazero.move_encoder import MoveEncoder

# --- Config Imports ---
from config import (
    # Project and File Paths
    PROJECT_NAME,
    TENSOR_CACHE_DIR,
    MODEL_DIR,
    BEST_MODEL_PATH,
    OPENINGS_SELFPLAY_PATH,
    # Self-Play Config
    DEFAULT_NUM_SELFPLAY_GAMES,
    NUM_SELFPLAY_WORKERS,
    MAX_FILES_IN_BUFFER,
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

try:
    SELFPLAY_OPENINGS = pd.read_csv(OPENINGS_SELFPLAY_PATH)
except FileNotFoundError:
    print(
        f"Warning: {OPENINGS_SELFPLAY_PATH} not found. Self-play will start from the standard position.",
        flush=True,
    )
    SELFPLAY_OPENINGS = None

# ─────────────────────────────────────────────────────────────────────────────
#  Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


def setup_selfplay_opening(env: ChessEnv, game_num: int = 0):
    """
    Sets up the opening position for a self-play game using the env's own method.
    Has a 75% chance to use the opening book, otherwise uses standard start.
    """
    prefix = f"[Worker {game_num + 1}]"
    opening_name = "Standard Opening"

    if (
        SELFPLAY_OPENINGS is not None
        and not SELFPLAY_OPENINGS.empty
        and random.random() < 0.75
    ):
        random_opening = SELFPLAY_OPENINGS.sample(n=1).iloc[0]
        move_string = random_opening["moves"]
        opening_name = random_opening.get("name", "Unknown Opening")

        env.set_board_from_moves(move_string)

    print(f"{prefix} Starting from opening: {opening_name}", flush=True)


def manage_replay_buffer():
    """
    Ensures the number of files in the replay buffer directory does not exceed a maximum.
    Deletes the oldest files if the limit is surpassed.
    """
    try:
        files = [
            os.path.join(TENSOR_CACHE_DIR, f)
            for f in os.listdir(TENSOR_CACHE_DIR)
            if f.endswith(".pt")
        ]
        if len(files) > MAX_FILES_IN_BUFFER:
            # Sort files by modification time (oldest first)
            files.sort(key=os.path.getmtime)

            num_to_delete = len(files) - MAX_FILES_IN_BUFFER
            for i in range(num_to_delete):
                os.remove(files[i])

            print(
                f"Replay buffer exceeded {MAX_FILES_IN_BUFFER} files. Deleted {num_to_delete} oldest files.",
                flush=True,
            )

    except Exception as e:
        print(
            f"Warning: Could not manage replay buffer. Error: {e}",
            flush=True,
        )


def choose_time_limit(prefix: str):
    """Randomly chooses a time limit to add variety to the training data."""
    if random.random() < 0.5:
        time_limit = random.uniform(0.5, 1.5)
    else:
        time_limit = random.uniform(2.0, 4.0)
    print(f"{prefix} Time limit per move: {time_limit:.1f}s", flush=True)
    return time_limit


def run_selfplay_worker(game_num: int) -> Tuple[List[dict], str, int, float]:
    """
    This is the target function for each worker process. It plays one game.
    """
    try:
        prefix = f"[Worker {game_num + 1}]"
        print(f"{prefix} Starting game...", flush=True)
        game_start_time = time.time()

        # Increase the dirichlet noise for 10% of games
        if game_num % 10 == 0:
            dirichlet_alpha = DIRICHLET_ALPHA * 2
            dirichlet_epsilon = DIRICHLET_EPSILON * 2
        else:
            dirichlet_alpha = DIRICHLET_ALPHA
            dirichlet_epsilon = DIRICHLET_EPSILON

        # Create a configuration for a self-play game
        config = GameConfig(
            white_model_path=BEST_MODEL_PATH,
            black_model_path=BEST_MODEL_PATH,
            time_limit=choose_time_limit(prefix),
            c_puct=SELFPLAY_CPUCT,
            use_dirichlet_noise=True,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            use_temperature_sampling=True,
            temp_threshold=TEMP_THRESHOLD,
            use_adjudication=True,  # Adjudication is typically not used for self-play
            adjudication_start_move=SELFPLAY_MAX_MOVES,
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

        game_positions, _, outcome_type, game_length, _ = play_game(config, env)

        if game_positions:
            encoder = MoveEncoder()
            for state_np, pi_np, z_value in game_positions:
                # Original
                state_tensor = torch.from_numpy(state_np.copy()).float()
                pi_tensor = torch.from_numpy(pi_np.copy()).float()
                z_tensor = torch.tensor([z_value], dtype=torch.float32)
                torch.save(
                    (state_tensor, pi_tensor, z_tensor),
                    f"{TENSOR_CACHE_DIR}/data_{uuid.uuid4()}.pt",
                )

                # Augmented (Flipped)
                flipped_state_np = np.ascontiguousarray(np.flip(state_np, axis=2))
                original_pi_torch = torch.from_numpy(pi_np)
                flipped_pi_torch = torch.zeros_like(original_pi_torch)
                for i, prob in enumerate(original_pi_torch):
                    if prob > 0:
                        flipped_idx = encoder.flip_map[i]
                        flipped_pi_torch[flipped_idx] = prob
                flipped_state_tensor = torch.from_numpy(flipped_state_np).float()
                torch.save(
                    (flipped_state_tensor, flipped_pi_torch, z_tensor),
                    f"{TENSOR_CACHE_DIR}/data_{uuid.uuid4()}.pt",
                )

        print(
            f"{prefix} Game finished. Outcome: {outcome_type}, Moves: {game_length}, Time: {format_time(time.time() - game_start_time)}",
            flush=True,
        )
        return outcome_type, game_length, config.time_limit
    except Exception as e:
        print(f"{prefix} Error: {e}", flush=True)
        return None, None, None


def log_selfplay_to_wandb(results, num_total_games):
    """
    Logs a final summary of the self-play session to Weights & Biases.
    """
    if not (wandb.run and not wandb.run.disabled):
        return  # Do nothing if wandb is disabled

    print("Logging self-play summary to Weights & Biases...", flush=True)

    summary_counts = {}
    game_lengths = []
    for outcome_type, game_length, _ in results:
        summary_counts[outcome_type] = summary_counts.get(outcome_type, 0) + 1
        game_lengths.append(game_length)

    avg_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0

    selfplay_summary = {
        "selfplay_total_games": num_total_games,
        "selfplay_avg_game_length": avg_length,
    }
    # Add the individual outcome counts to the summary
    selfplay_summary.update({f"outcome_{k}": v for k, v in summary_counts.items()})

    wandb.log(selfplay_summary)
    print("Finished logging to Weights & Biases.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Main Execution Block
# ─────────────────────────────────────────────────────────────────────────────


def main(args):
    """Orchestrates the self-play data generation process using a pool of workers."""

    print(
        f"Starting self-play for {args.num_games} games using {NUM_SELFPLAY_WORKERS} parallel workers...",
        flush=True,
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
        f"\nAll self-play games finished in {format_time(time.time() - start_time)}. Processing results...",
        flush=True,
    )

    log_selfplay_to_wandb(results, args.num_games)

    manage_replay_buffer()

    wandb.finish()


if __name__ == "__main__":
    ensure_project_root()

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
    args = parser.parse_args()
    main(args)
