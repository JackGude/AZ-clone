# eval.py

import argparse
from itertools import repeat
import json
import multiprocessing
import os
import random
import sys
import time

from alphazero.env import ChessEnv
from alphazero.game_runner import GameConfig, play_game
from alphazero.utils import ensure_project_root, format_time
from chess import pgn
from chess.pgn import StringExporter
from config import (
    ADJUDICATION_START_MOVE,
    BEST_MODEL_PATH,
    CANDIDATE_MODEL_PATH,
    DEFAULT_EVAL_CPUCT,
    DEFAULT_EVAL_TIME_LIMIT,
    DEFAULT_NUM_EVAL_GAMES,
    DRAW_ADJUDICATION_PATIENCE,
    DRAW_ADJUDICATION_THRESHOLD,
    EVAL_GAMES_DIR,
    NUM_EVAL_WORKERS,
    OPENINGS_EVAL_PATH,
    PROJECT_NAME,
)
import pandas as pd
import wandb


# ─────────────────────────────────────────────────────────────────────────────
#  Load Evaluation Openings
# ─────────────────────────────────────────────────────────────────────────────
try:
    EVALUATION_OPENINGS = pd.read_csv(OPENINGS_EVAL_PATH)
except FileNotFoundError:
    print(f"Warning: {OPENINGS_EVAL_PATH} not found.", flush=True)
    EVALUATION_OPENINGS = None


# ─────────────────────────────────────────────────────────────────────────────
#  Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


def run_eval_worker(
    game_idx,
    old_model_path,
    new_model_path,
    time_limit,
    num_total_games,
    num_unique_openings,
    openings,
    gen_id,
):
    """
    This is the target function for each worker process. It plays one evaluation game.
    """

    # --- Create a prefix for logging that matches the self-play format ---
    prefix = f"[Worker {game_idx + 1}]"

    # --- Determine which model plays which color for this game ---
    is_new_white = game_idx < num_unique_openings
    if is_new_white:
        white_path = new_model_path
        black_path = old_model_path
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

    # --- Setup the environment with the correct opening moves ---
    env = ChessEnv()
    opening_name, opening_moves = openings[game_idx % num_unique_openings]
    env.set_board_from_moves(opening_moves)

    # --- Create a configuration for an evaluation game ---
    config = GameConfig(
        white_model_path=white_path,
        black_model_path=black_path,
        time_limit=time_limit,
        c_puct=DEFAULT_EVAL_CPUCT,
        use_dynamic_cpuct=False,
        use_dirichlet_noise=False,
        dirichlet_alpha=0,
        dirichlet_epsilon=0,
        use_temperature_sampling=False,
        temp_threshold=0,
        use_adjudication=True,
        adjudication_start_move=ADJUDICATION_START_MOVE,
        draw_adjudication_threshold=DRAW_ADJUDICATION_THRESHOLD,
        draw_adjudication_patience=DRAW_ADJUDICATION_PATIENCE,
        resign_threshold=1.1,
        selfplay_max_moves=1000,
        verbose=False,
        log_prefix=prefix,
    )

    # --- Create a PGN file for the game ---
    game = pgn.Game()
    game.setup(env.board)
    game.headers["Event"] = f"Evaluation Match Gen {gen_id}"
    game.headers["Site"] = "Local"
    game.headers["Date"] = str(time.strftime("%Y.%m.%d"))
    game.headers["Round"] = str(game_idx + 1)
    game.headers["White"] = "New Model" if is_new_white else "Old Model"
    game.headers["Black"] = "Old Model" if is_new_white else "New Model"
    game.headers["Opening"] = opening_name

    # --- Play the game, passing the PGN node to be updated live ---
    _, outcome_type, move_count = play_game(config, env=env, pgn_node=game.end())

    # --- Save the now-complete PGN file ---
    if outcome_type in ["checkmate_white", "resign_black"]:
        game.headers["Result"] = "1-0"  # White wins by checkmate or black resigns
    elif outcome_type in ["checkmate_black", "resign_white"]:
        game.headers["Result"] = "0-1"  # Black wins by checkmate or white resigns
    else:  # draw_cap, draw_adjudicated, draw_game
        game.headers["Result"] = "1/2-1/2"

    os.makedirs(os.path.join(EVAL_GAMES_DIR, gen_id), exist_ok=True)
    pgn_path = os.path.join(
        EVAL_GAMES_DIR, gen_id, f"game_{game_idx + 1}_{outcome_type}.pgn"
    )

    try:
        # Use the dedicated StringExporter for robust writing
        exporter = StringExporter(headers=True, variations=True, comments=True)
        pgn_string = game.accept(exporter)
        with open(pgn_path, "w", encoding="utf-8") as f:
            f.write(pgn_string)
    except Exception as e:
        print(
            f"{prefix} Failed to write PGN file {pgn_path}. Error: {e}",
            file=sys.stderr,
            flush=True,
        )

    print(
        f"{prefix} Game finished. Outcome: {outcome_type}, Moves: {move_count}",
        flush=True,
    )

    # --- Convert outcome to a score from the NEW model's perspective. ---
    if outcome_type in ["checkmate_white", "resign_black"]:  # White wins
        return 1.0 if is_new_white else -1.0
    elif outcome_type in ["checkmate_black", "resign_white"]:  # Black wins
        return -1.0 if is_new_white else 1.0
    return 0.0  # draw_cap, draw_adjudicated, draw_game


def log_evaluation_to_wandb(
    results, win_rate, wins, losses, draws, num_unique_openings, openings
):
    """
    Logs detailed per-game data and a final summary for the evaluation match
    to Weights & Biases.
    """
    if not (wandb.run and not wandb.run.disabled):
        return  # Do nothing if wandb is disabled

    print("Logging results to Weights & Biases...", flush=True)

    # --- Create and log a detailed table of each game's result ---
    eval_table = wandb.Table(
        columns=["Game Index", "New Model Color", "Opening", "Result"]
    )

    for i, score in enumerate(results):
        if i < num_unique_openings:
            new_model_color = "White"
        else:
            new_model_color = "Black"

        opening_name, _ = openings[i % num_unique_openings]
        result_str = "Win" if score == 1.0 else "Loss" if score == -1.0 else "Draw"

        eval_table.add_data(i + 1, new_model_color, opening_name, result_str)

    # --- Log the summary dictionary, including the results table ---
    eval_summary = {
        "eval_win_rate": win_rate,
        "eval_wins": wins,
        "eval_draws": draws,
        "eval_losses": losses,
        "evaluation_games_table": eval_table,
    }

    wandb.log(eval_summary)
    print("Finished logging to Weights & Biases.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Main Execution Block
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(
    checkpoint_old,
    checkpoint_new,
    num_games,
    time_limit,
    num_workers,
    result_file,
    gen_id,
):
    """
    Orchestrates a head-to-head evaluation using a pool of parallel workers.
    """
    if EVALUATION_OPENINGS is None or EVALUATION_OPENINGS.empty:
        print(
            "Error: Evaluation openings are not loaded. Cannot start evaluation.",
            file=sys.stderr,
            flush=True,
        )
        return
    else:
        # Convert DataFrame to list of tuples (opening_name, opening_moves)
        openings = list(zip(EVALUATION_OPENINGS["name"], EVALUATION_OPENINGS["moves"]))
        random.shuffle(openings)

    if num_games % 2 != 0:
        print(
            "Error: Number of games must be even. Rounding up to the nearest even number.",
            file=sys.stderr,
            flush=True,
        )
        num_games += 1

    num_unique_openings = min(len(openings), num_games // 2)

    print(
        f"\n--- Starting evaluation: {num_games} games, {time_limit}s per move, {num_workers} workers ---",
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
        repeat(num_unique_openings),
        repeat(openings),
        repeat(gen_id),
    )

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.starmap(run_eval_worker, tasks)

    print(
        f"\nAll evaluation games finished in {format_time(time.time() - total_start_time)}. Processing results...",
        file=sys.stderr,
        flush=True,
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

    log_evaluation_to_wandb(
        results, win_rate, wins, losses, draws, num_unique_openings, openings
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
        gen_id=args.gen_id,
    )

    wandb.finish()


if __name__ == "__main__":
    ensure_project_root()
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
        help=f"Number of games to play. Defaults to {DEFAULT_NUM_EVAL_GAMES}. Must be even.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=DEFAULT_EVAL_TIME_LIMIT,
        help=f"Time limit in seconds per move. Defaults to {DEFAULT_EVAL_TIME_LIMIT}.",
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
