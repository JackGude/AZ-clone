# evaluate.py

import argparse
import time
import json
import wandb
import sys
import multiprocessing
from itertools import repeat
from automate import format_time

# --- Local Project Imports ---
from alphazero.game_runner import GameConfig, play_game
from alphazero.env import ChessEnv
from alphazero.utils import EVALUATION_OPENINGS  # <-- Import from utils

# --- Config Imports ---
from config import (
    # Project and File Paths
    PROJECT_NAME,
    BEST_MODEL_PATH,
    CANDIDATE_MODEL_PATH,
    # Evaluation Config
    NUM_EVAL_WORKERS,
    DEFAULT_NUM_EVAL_GAMES,
    DEFAULT_EVAL_TIME_LIMIT,
    DEFAULT_EVAL_CPUCT,
    DRAW_ADJUDICATION_THRESHOLD,
    DRAW_ADJUDICATION_PATIENCE,
    WIN_ADJUDICATION_THRESHOLD,
    WIN_ADJUDICATION_PATIENCE,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


def run_eval_worker(game_idx, old_model_path, new_model_path, time_limit, num_total_games):
    """
    This is the target function for each worker process. It plays one evaluation game.
    """

    # Create a prefix for logging that matches the self-play format
    prefix = f"[Worker {game_idx + 1}]"

    # Determine which model plays which color for this game
    is_new_white = game_idx % 2 == 0
    if is_new_white:
        white_path = new_model_path
        black_path = old_model_path
        is_new_white = True
        print(
            f"{prefix} Game {game_idx + 1}/{num_total_games}... (New plays as White)",
            file=sys.stderr,
            flush=True,
        )
    else:
        white_path = old_model_path
        black_path = new_model_path
        print(
            f"{prefix} Game {game_idx + 1}/{num_total_games}... (New plays as Black)",
            file=sys.stderr,
            flush=True,
        )

    # Setup the environment with the correct opening FEN
    env = ChessEnv()
    opening_name, opening_fen = EVALUATION_OPENINGS[game_idx % len(EVALUATION_OPENINGS)]
    env.set_board_from_fen(opening_fen)
    print(
        f"{prefix} Starting with opening: {opening_name}", file=sys.stderr, flush=True
    )

    # Create a configuration for an evaluation game
    config = GameConfig(
        white_model_path=white_path,
        black_model_path=black_path,
        time_limit=time_limit,
        c_puct=DEFAULT_EVAL_CPUCT,
        use_dirichlet_noise=False,
        dirichlet_alpha=0,
        dirichlet_epsilon=0,
        use_temperature_sampling=False,
        temp_threshold=0,
        use_adjudication=True,
        win_adjudication_threshold=WIN_ADJUDICATION_THRESHOLD,
        win_adjudication_patience=WIN_ADJUDICATION_PATIENCE,
        draw_adjudication_threshold=DRAW_ADJUDICATION_THRESHOLD,
        draw_adjudication_patience=DRAW_ADJUDICATION_PATIENCE,
        resign_threshold=1.1,
        selfplay_max_moves=1000,
        verbose=True,
        log_prefix=prefix,
    )

    _, numerical_outcome, outcome_type, move_count = play_game(config, env=env)

    print(
        f"{prefix} Game finished. Outcome: {outcome_type}, Moves: {move_count}",
        flush=True,
    )

    # Convert outcome to a score from the NEW model's perspective.
    if is_new_white:
        return numerical_outcome
    else:
        return -numerical_outcome


# ─────────────────────────────────────────────────────────────────────────────
#  Main Execution Block
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(
    checkpoint_old, checkpoint_new, num_games, time_limit, num_workers, result_file
):
    """
    Orchestrates a head-to-head match using a pool of parallel workers.
    """
    if not EVALUATION_OPENINGS:
        print(
            "Error: Evaluation openings are not loaded. Cannot start match.",
            file=sys.stderr,
        )
        return

    print(
        f"\n--- Starting match: {num_games} games, {time_limit}s per move, {num_workers} workers ---",
        file=sys.stderr,
        flush=True,
    )
    total_start_time = time.time()

    # Prepare the list of tasks for the workers
    tasks = zip(
        range(num_games),
        repeat(checkpoint_old),
        repeat(checkpoint_new),
        repeat(time_limit),
        repeat(num_games),
    )

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.starmap(run_eval_worker, tasks)

    print(
        f"\nAll evaluation games finished in {format_time(time.time() - total_start_time)}. Processing results..."
    )

    wins = sum(1 for r in results if r == 1.0)
    draws = sum(1 for r in results if r == 0.0)
    losses = sum(1 for r in results if r == -1.0)

    win_rate = (wins + 0.5 * draws) / num_games if num_games > 0 else 0.0

    print("\n=== Evaluation Summary ===", file=sys.stderr, flush=True)
    print(
        f"Results for NEW model vs OLD model: {wins} Wins, {losses} Losses, {draws} Draws",
        file=sys.stderr,
        flush=True,
    )
    print(f"Win Rate of NEW model: {win_rate * 100:.1f}%", file=sys.stderr, flush=True)

    if wandb.run:
        wandb.log(
            {
                "eval_win_rate": win_rate,
                "eval_wins": wins,
                "eval_draws": draws,
                "eval_losses": losses,
            }
        )

    with open(result_file, "w") as f:
        json.dump({"win_rate": win_rate}, f)


def main(args):
    """Main function for standalone execution."""
    wandb.init(
        project=PROJECT_NAME,
        group=args.gen_id,
        name=f"{args.gen_id}-evaluation",
        job_type="evaluation",
        mode="disabled" if args.no_wandb else "online",
    )

    evaluate(
        checkpoint_old=args.old,
        checkpoint_new=args.new,
        num_games=args.games,
        time_limit=args.time_limit,
        num_workers=NUM_EVAL_WORKERS,
        result_file=args.result_file,
    )

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate two AlphaZeroNet checkpoints head-to-head."
    )
    parser.add_argument(
        "--old",
        type=str,
        default=BEST_MODEL_PATH,
        help="Path to the 'old' incumbent checkpoint. Defaults to the best model.",
    )
    parser.add_argument(
        "--new",
        type=str,
        default=CANDIDATE_MODEL_PATH,
        help="Path to the 'new' challenger checkpoint. Defaults to the candidate model.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=DEFAULT_NUM_EVAL_GAMES,
        help="Number of games to play. Defaults to 24.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=DEFAULT_EVAL_TIME_LIMIT,
        help="Time limit in seconds per move. Defaults to 10.",
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable wandb logging."
    )
    parser.add_argument(
        "--gen-id", type=str, default="manual", help="Generation ID for this run."
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default="logs/eval_results.json",
        help="Path to write the evaluation result JSON file.",
    )
    args = parser.parse_args()
    main(args)
