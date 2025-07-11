# play.py

import argparse
import multiprocessing
import os
import random
import sys
import time
from typing import Tuple
import uuid

from alphazero.env import ChessEnv
from alphazero.game_runner import GameConfig, play_game
from alphazero.move_encoder import MoveEncoder
from alphazero.utils import ensure_project_root, format_time, load_or_initialize_model
from config import (
    BEST_MODEL_PATH,
    DEFAULT_NUM_SELFPLAY_GAMES,
    DIRICHLET_ALPHA,
    DIRICHLET_EPSILON,
    DIRICHLET_MODULO,
    DRAW_CACHE_DIR,
    MAX_DECISIVE_GAMES,
    MAX_DRAW_GAMES,
    MODEL_DIR,
    NUM_SELFPLAY_WORKERS,
    OPENINGS_SELFPLAY_PATH,
    PAST_CHAMPS_DIR,
    PROJECT_NAME,
    RESIGN_THRESHOLD,
    SELFPLAY_CPUCT,
    SELFPLAY_MAX_MOVES,
    SPARRING_PARTNER_PROB,
    TEMP_THRESHOLD,
    WIN_CACHE_DIR,
)
import numpy as np
import pandas as pd
import torch
import wandb

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Ensure necessary directories exist, paths come from config.py
os.makedirs(WIN_CACHE_DIR, exist_ok=True)
os.makedirs(DRAW_CACHE_DIR, exist_ok=True)
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
    caches = [(WIN_CACHE_DIR, MAX_DECISIVE_GAMES), (DRAW_CACHE_DIR, MAX_DRAW_GAMES)]

    for cache_dir, max_files in caches:
        try:
            files = [
                os.path.join(cache_dir, f)
                for f in os.listdir(cache_dir)
                if f.endswith(".pt")
            ]
            if len(files) > max_files:
                # Sort files by modification time (oldest first)
                files.sort(key=os.path.getmtime)

                num_to_delete = len(files) - max_files
                for i in range(num_to_delete):
                    os.remove(files[i])

        except Exception as e:
            print(
                f"Warning: Could not manage replay buffer at {cache_dir}. Error: {e}",
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


def run_selfplay_worker(game_num: int) -> Tuple[str, int, float]:
    """
    This is the target function for each worker process. It plays one game,
    saves the training data (including legality masks), and returns summary stats.
    
    With probability SPARRING_PARTNER_PROB, the black player will be a random past champion
    instead of the current best model, to provide more diverse training.
    """
    prefix = f"[Worker {game_num + 1}]"

    try:
        # print(f"{prefix} Starting game...", flush=True)
        game_start_time = time.time()

        # Increase the dirichlet noise for a percentage of games
        if game_num % DIRICHLET_MODULO == 0:
            dirichlet_alpha = DIRICHLET_ALPHA * 2
            dirichlet_epsilon = min(0.5, DIRICHLET_EPSILON * 2)
        else:
            dirichlet_alpha = DIRICHLET_ALPHA
            dirichlet_epsilon = DIRICHLET_EPSILON

        # Determine model paths - sometimes use a past champion as sparring partner
        white_model_path = BEST_MODEL_PATH
        black_model_path = BEST_MODEL_PATH
        
        # With probability SPARRING_PARTNER_PROB, choose a random past champion as the opponent
        if random.random() < SPARRING_PARTNER_PROB and os.path.exists(PAST_CHAMPS_DIR):
            past_champs = [f for f in os.listdir(PAST_CHAMPS_DIR) if f.endswith('.pth')]
            if past_champs:
                chosen_champ = random.choice(past_champs)
                black_model_path = os.path.join(PAST_CHAMPS_DIR, chosen_champ)
                print(f"{prefix} Using past champion as sparring partner: {chosen_champ}", flush=True)

        # Create a configuration for a self-play game
        config = GameConfig(
            white_model_path=white_model_path,
            black_model_path=black_model_path,
            time_limit=choose_time_limit(prefix),
            c_puct=SELFPLAY_CPUCT,
            use_dynamic_cpuct=True,
            use_dirichlet_noise=True,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            use_temperature_sampling=True,
            temp_threshold=TEMP_THRESHOLD,
            use_adjudication=False,
            adjudication_start_move=0,
            draw_adjudication_threshold=0.0,
            draw_adjudication_patience=1,
            resign_threshold=RESIGN_THRESHOLD,
            selfplay_max_moves=SELFPLAY_MAX_MOVES,
            verbose=False,
            log_prefix=prefix,
        )

        env = ChessEnv()
        setup_selfplay_opening(env, game_num)

        # --- Play the Game ---
        # The game_positions now contains dictionaries with the legality_mask
        game_positions, outcome_type, game_length = play_game(
            config, env
        )

        # --- Process and Save Training Tensors ---
        if game_positions:
            encoder = MoveEncoder()
            cache_dir = DRAW_CACHE_DIR if "draw" in outcome_type else WIN_CACHE_DIR
            os.makedirs(cache_dir, exist_ok=True)

            # Initialize lists to store all positions from this game
            game_chunk = []
            flipped_game_chunk = []

            # Process all positions first
            for pos_data in game_positions:
                # Unpack the dictionary from game_runner
                state_np = pos_data["state_history"]
                pi_np = pos_data["pi"]
                legality_np = pos_data["legality_mask"]
                z_value = pos_data["z_value"]

                # --- Original Data Point ---
                state_tensor = torch.from_numpy(np.array(state_np)).float()
                pi_tensor = torch.from_numpy(pi_np).float()
                z_tensor = torch.tensor([z_value], dtype=torch.float32)
                legality_tensor = torch.from_numpy(legality_np).float()

                # Add to game chunk
                game_chunk.append((state_tensor, pi_tensor, z_tensor, legality_tensor))

                # --- Create Augmented (Flipped) Data Point ---
                flipped_state_np = np.ascontiguousarray(np.flip(state_np, axis=2))

                original_pi_torch = torch.from_numpy(pi_np)
                flipped_pi_torch = torch.zeros_like(original_pi_torch)
                for i, prob in enumerate(original_pi_torch):
                    if prob > 0:
                        flipped_idx = encoder.flip_map[i]
                        flipped_pi_torch[flipped_idx] = prob

                # Flip the legality mask as well
                flipped_legality_np = np.zeros_like(legality_np)
                for i, is_legal in enumerate(legality_np):
                    if is_legal > 0:
                        flipped_idx = encoder.flip_map[i]
                        flipped_legality_np[flipped_idx] = 1.0

                flipped_state_tensor = torch.from_numpy(flipped_state_np).float()
                flipped_legality_tensor = torch.from_numpy(flipped_legality_np).float()

                # Add to flipped game chunk
                flipped_game_chunk.append(
                    (
                        flipped_state_tensor,
                        flipped_pi_torch,
                        z_tensor,
                        flipped_legality_tensor,
                    )
                )

            # Save the complete game as a single chunk
            if game_chunk:
                # Save original game
                torch.save(game_chunk, f"{cache_dir}/chunk_{uuid.uuid4()}.pt")
                # Save flipped version
                torch.save(flipped_game_chunk, f"{cache_dir}/chunk_{uuid.uuid4()}.pt")

        print(
            f"{prefix} Game finished. Outcome: {outcome_type}, Moves: {game_length}, Time: {format_time(time.time() - game_start_time)}",
            flush=True,
        )
        return outcome_type, game_length, config.time_limit

    except Exception as e:
        print(f"{prefix} CRITICAL WORKER ERROR: {e}", file=sys.stderr, flush=True)
        import traceback

        traceback.print_exc()
        return "error", 0, 0


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
    
    # Calculate percentages
    total_games = sum(summary_counts.values())
    drawn_games = sum(count for outcome, count in summary_counts.items() if 'draw' in outcome.lower())
    win_loss_games = total_games - drawn_games
    
    drawn_percentage = (drawn_games / total_games * 100) if total_games > 0 else 0
    win_loss_percentage = (win_loss_games / total_games * 100) if total_games > 0 else 0

    selfplay_summary = {
        "selfplay_total_games": num_total_games,
        "selfplay_avg_game_length": avg_length,
        "selfplay_drawn_percentage": drawn_percentage,
        "selfplay_win_loss_percentage": win_loss_percentage,
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

    # Initialize a fresh model if necessary
    if not os.path.exists(BEST_MODEL_PATH):
        load_or_initialize_model(BEST_MODEL_PATH, verbose=True)

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
