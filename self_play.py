# self_play.py

import os
import random
import pickle
import torch
import chess
import numpy as np
import time
import json
import uuid
import argparse
import wandb
import re
from datetime import datetime
from alphazero.env import ChessEnv
from alphazero.move_encoder import MoveEncoder
from alphazero.state_encoder import encode_history
from alphazero.mcts import MCTS
from alphazero.model import AlphaZeroNet
from automate import format_time
from config import (
    # Project and File Paths
    PROJECT_NAME,
    REPLAY_DIR,
    CHECKPOINT_DIR,
    LOGS_DIR,
    BEST_CHECKPOINT,
    OPENINGS_SELFPLAY_PATH,
    # Self-Play Config
    DEFAULT_NUM_SELFPLAY_GAMES,
    MAX_GAMES_IN_BUFFER,
    SELFPLAY_CPUCT,
    DIRICHLET_EPSILON,
    DIRICHLET_ALPHA,
    TEMP_THRESHOLD,
    SELFPLAY_MAX_MOVES,
    RESIGN_THRESHOLD,
    DRAW_CAP_PENALTY
)
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Load openings from CSV
try:
    SELFPLAY_OPENINGS = pd.read_csv(OPENINGS_SELFPLAY_PATH)
    print(f"Successfully loaded {len(SELFPLAY_OPENINGS)} openings for self-play.")
except FileNotFoundError:
    print(
        f"Warning: {OPENINGS_SELFPLAY_PATH} not found. Self-play will start from the standard position."
    )
    SELFPLAY_OPENINGS = None

# Ensure necessary directories exist, paths come from config.py
os.makedirs(REPLAY_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Game Generation Logic
# ─────────────────────────────────────────────────────────────────────────────

def get_resume_game_count(log_path):
    """
    Parses a self-play log file to find the last game number that started.
    """
    last_game_started = 0
    if not os.path.exists(log_path):
        return 0 # No log file means we start from scratch

    # The regex looks for the pattern "=== Starting game X/Y ===" and captures X
    pattern = re.compile(r"=== Starting game (\d+)")
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    # Continuously update with the latest game number found
                    last_game_started = int(match.group(1))
    except Exception as e:
        print(f"Warning: Could not parse log file {log_path} for resuming. Starting fresh. Error: {e}")
        return 0
        
    print(f"\nResuming self-play. Found {last_game_started} games already started in log file.\n", flush=True)
    return last_game_started

def get_board_from_moves(move_string):
    """
    Takes a space-separated string of moves in SAN, plays them on a board,
    and returns the final chess.Board object with a full move history.
    """
    board = chess.Board()
    try:
        for move in move_string.split(" "):
            board.push_san(move)
        return board
    except (ValueError, IndexError):
        return chess.Board()


def setup_opening(env):
    """
    Sets up the opening position for a self-play game.
    Has a 75% chance to use the opening book, otherwise uses standard start.
    """
    opening_name = "Standard Opening"
    if (
        SELFPLAY_OPENINGS is not None
        and not SELFPLAY_OPENINGS.empty
        and random.random() < 0.75
    ):
        random_opening = SELFPLAY_OPENINGS.sample(n=1).iloc[0]
        move_string = random_opening["moves"]
        starting_board = get_board_from_moves(move_string)
        env.board = starting_board  # Set the environment's board directly
        opening_name = random_opening.get("name", "Unknown Opening")
    print(f"--- Starting from opening: {opening_name} ---", flush=True)


def choose_time_limit():
    """Randomly chooses a time limit to add variety to the training data."""
    if random.random() < 0.5:
        time_limit = random.uniform(0.5, 1.5)
    else:
        time_limit = random.uniform(2.0, 4.0)
    print(f"Time limit per move: {time_limit:.1f}s", flush=True)
    return time_limit


def self_play_game(net, encoder, device):
    """
    Plays one full game of chess via self-play using MCTS.
    Returns a list of training examples, the outcome type, and game length.
    """
    env = ChessEnv()
    env.reset()

    setup_opening(env)

    time_limit = choose_time_limit()

    mcts = MCTS(
        net,
        encoder,
        time_limit=time_limit,
        c_puct=SELFPLAY_CPUCT,
        device=device,
        batch_size=64,
        dirichlet_alpha=DIRICHLET_ALPHA,
        dirichlet_epsilon=DIRICHLET_EPSILON,
    )

    game_records = []  # Stores (history, pi_vector, player_turn) for each move
    move_number = 0
    start_time = time.time()

    # Main game loop
    while True:
        # Check for game length cap to prevent infinitely long games
        if move_number >= SELFPLAY_MAX_MOVES:
            outcome, outcome_type = DRAW_CAP_PENALTY, "draw_cap"
            break

        move_number += 1

        # Only apply Dirichlet noise for early moves to encourage opening exploration.
        mcts.dirichlet_alpha = DIRICHLET_ALPHA if move_number <= TEMP_THRESHOLD else 0.0

        # Run the MCTS search
        root, _ = mcts.run(env)

        # Resignation logic based on the post-search Q-value
        if root.Q < -RESIGN_THRESHOLD:
            to_move = env.board.turn
            outcome = (
                -1.0 if to_move == chess.WHITE else 1.0
            )  # The resigning player loses
            outcome_type = "resign_white" if to_move == chess.WHITE else "resign_black"
            break

        # Construct the policy vector (pi) from the MCTS visit counts
        counts = torch.zeros(encoder.mapping_size, dtype=torch.float32)
        for mv, child in root.children.items():
            counts[encoder.encode(mv)] = child.N

        pi = (
            (counts / counts.sum()).numpy()
            if counts.sum() > 0
            else np.zeros(encoder.mapping_size, dtype=np.float32)
        )

        # Record the state, policy, and player turn for later training example creation
        game_records.append((list(env.history), pi, env.board.turn))

        # Temperature-based move selection
        if move_number < TEMP_THRESHOLD:
            # For early moves, sample from the policy to create varied games
            move_idx = int(np.random.choice(len(pi), p=pi))
        else:
            # For later moves, play greedily by selecting the most visited move
            move_idx = int(counts.argmax().item())

        move = encoder.decode(move_idx)
        _, reward, done = env.step(move)

        # Check for game termination (checkmate or draw)
        if done:
            outcome = reward
            if reward == 1.0:
                outcome_type = "checkmate_white"
            elif reward == -1.0:
                outcome_type = "checkmate_black"
            else:
                outcome_type = "draw_game"
            break

    print(
        f"Game finished. Moves: {move_number}. Outcome: {outcome_type}. Time: {format_time(time.time() - start_time)}",
        flush=True,
    )

    # After the game, prepare the training examples
    examples = []
    for hist, pi_vec, player_turn in game_records:
        # The value 'z' must be from the perspective of the player at that state
        z_value = outcome if player_turn == chess.WHITE else -outcome
        state_tensor = encode_history(hist)
        state_np = state_tensor.numpy()
        examples.append((state_np, pi_vec, z_value))

    return examples, outcome_type, move_number


def manage_replay_buffer():
    """
    Keeps a maximum number of recent DRAW games, while preserving all WIN/LOSS games.
    """
    try:
        all_game_files = [os.path.join(REPLAY_DIR, f) for f in os.listdir(REPLAY_DIR) if f.endswith('.pkl')]
        draw_files = [f for f in all_game_files if "draw" in os.path.basename(f).lower()]
        draw_files.sort(key=os.path.getmtime)
        
        while len(draw_files) > MAX_GAMES_IN_BUFFER:
            file_to_delete = draw_files.pop(0)
            os.remove(file_to_delete)
    except Exception as e:
        print(f"Warning: Could not manage replay buffer. Error: {e}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Main Execution Block
# ─────────────────────────────────────────────────────────────────────────────


def main(args):
    """Orchestrates the self-play data generation process."""

    # Initialize wandb with the new format
    wandb.init(
        project=PROJECT_NAME,
        group=args.gen_id,
        name=f"{args.gen_id}-self-play",
        job_type="self-play",
        mode="disabled" if args.no_wandb else "online",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the best available model checkpoint
    net = AlphaZeroNet(in_channels=119).to(device)
    checkpoint_path = BEST_CHECKPOINT

    if os.path.exists(checkpoint_path):
        net.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        print(f"Loaded best weights from '{checkpoint_path}'", flush=True)
    else:
        print(
            f"No checkpoint found at '{checkpoint_path}'; using randomly initialized model.",
            flush=True,
        )
    net.eval()

    log_path = os.path.join(LOGS_DIR, f"{args.gen_id}-self-play.log")
    games_already_run = get_resume_game_count(log_path)
    
    games_to_play = args.num_games - games_already_run

    if games_to_play <= 0:
        print("Self-play for this generation is already complete. Exiting.")
        if wandb.run:
            wandb.run.finish()
        return

    encoder = MoveEncoder()
    total_start_time = time.time()
    summary_counts = {
        k: 0
        for k in [
            "resign_white",
            "resign_black",
            "checkmate_white",
            "checkmate_black",
            "draw_game",
            "draw_cap",
        ]
    }
    game_lengths = []

    for i in range(games_to_play):
        current_game_number = games_already_run + i + 1
        print(f"\n=== Starting game {current_game_number}/{args.num_games} ===", flush=True)

        game_examples, outcome_type, game_length = self_play_game(
            net, encoder, device=device
        )

        # Collect stats for logging
        if outcome_type:
            summary_counts[outcome_type] += 1
        game_lengths.append(game_length)

        # Save this game's data to its own unique file
        if game_examples:
            game_id = uuid.uuid4()
            game_path = os.path.join(REPLAY_DIR, f"game_{outcome_type}_{game_id}.pkl")
            with open(game_path, "wb") as f:
                pickle.dump(game_examples, f)

        manage_replay_buffer()

    # --- Log summary statistics to wandb ---
    total_selfplay_time = time.time() - total_start_time
    avg_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0

    selfplay_summary = {
        "selfplay_total_games": args.num_games,
        "total_selfplay_time": total_selfplay_time,
        "selfplay_avg_game_length": avg_length,
    }
    for k, v in summary_counts.items():
        selfplay_summary[f"outcome_{k}"] = v

    wandb.log(selfplay_summary)

    # Write results to JSON file
    with open(args.result_file, 'w') as f:
        json.dump(selfplay_summary, f, indent=2)

    # Finish the wandb run
    if wandb.run:
        wandb.run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run self-play games to generate training data."
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=DEFAULT_NUM_SELFPLAY_GAMES,
        help="Number of self-play games to generate.",
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable wandb logging for this run."
    )
    parser.add_argument(
        "--gen-id", type=str, required=True, help="Generation ID for this run."
    )
    parser.add_argument(
        "--result-file", type=str, required=True, help="Path to write the self-play results JSON file"
    )
    args = parser.parse_args()
    main(args)
