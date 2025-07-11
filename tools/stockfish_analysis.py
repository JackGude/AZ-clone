# stockfish_analysis.py (Refactored)
import argparse
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
import os

from tqdm import tqdm

import chess
import chess.engine
import chess.pgn
from config import EVAL_GAMES_DIR, NUM_WORKERS, PROJECT_NAME, STOCKFISH_PATH

# --- Analysis Constants ---
ANALYSIS_DEPTH = 14
MATE_SCORE = 10000
BLUNDER_THRESHOLD = 100
MISTAKE_THRESHOLD = 50
WINNING_ADVANTAGE = 200  # in centipawns

# --- Worker-specific global engine ---
engine = None


# --- Data Structures ---
@dataclass
class PlayerStats:
    """Stores analysis metrics for a single player in one game."""

    player_name: str
    total_cp_loss: float = 0.0
    move_count: int = 0
    blunders: int = 0
    mistakes: int = 0
    was_winning: bool = False
    was_losing: bool = False
    # Fields to be added after analysis
    color: str = ""
    final_result: str = ""
    opening: str = ""


# --- Core Analysis Logic ---


def init_worker(stockfish_path: str) -> None:
    """Initializes a Stockfish engine for each worker process."""
    global engine
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except Exception as e:
        print(f"Worker {os.getpid()} failed to init engine: {e}")
        engine = None


def analyze_game(pgn_path: str) -> dict | None:
    """Analyzes a single PGN file and returns a dictionary of PlayerStats."""
    if engine is None:
        return None

    try:
        with open(pgn_path, "r") as f:
            game = chess.pgn.read_game(f)
        if game is None:
            return None
    except Exception:
        return None

    white_stats = PlayerStats(player_name=game.headers.get("White", "Unknown"))
    black_stats = PlayerStats(player_name=game.headers.get("Black", "Unknown"))

    board = game.board()
    limit = chess.engine.Limit(depth=ANALYSIS_DEPTH)

    try:
        info = engine.analyse(board, limit)
        prev_score = info["score"].white().score(mate_score=MATE_SCORE)

        for node in game.mainline():
            is_white_move = board.turn == chess.WHITE
            score_before = prev_score
            board.push(node.move)
            info_after = engine.analyse(board, limit)
            score_after = info_after["score"].white().score(mate_score=MATE_SCORE)

            current_player_stats = white_stats if is_white_move else black_stats
            cp_loss = (
                (score_before - score_after)
                if is_white_move
                else (score_after - score_before)
            )

            current_player_stats.total_cp_loss += cp_loss
            current_player_stats.move_count += 1
            if cp_loss > BLUNDER_THRESHOLD:
                current_player_stats.blunders += 1
            elif cp_loss > MISTAKE_THRESHOLD:
                current_player_stats.mistakes += 1

            if score_before > WINNING_ADVANTAGE:
                white_stats.was_winning = True
                black_stats.was_losing = True
            elif score_before < -WINNING_ADVANTAGE:
                black_stats.was_winning = True
                white_stats.was_losing = True

            prev_score = score_after

    except chess.engine.EngineTerminatedError:
        return None

    # Add post-analysis metadata
    opening = game.headers.get("Opening", "Unknown")
    result = game.headers.get("Result", "*")
    for stats, color in [(white_stats, "White"), (black_stats, "Black")]:
        stats.opening = opening
        stats.final_result = result
        stats.color = color

    return {"white": white_stats, "black": black_stats}


# --- Report Generation Logic ---


def run_analysis_pool(pgn_files: list[str], num_workers: int) -> list[dict]:
    """Runs the game analysis in a multiprocessing pool."""
    print(
        f"Found {len(pgn_files)} games. Analyzing with {num_workers} workers (Depth: {ANALYSIS_DEPTH})..."
    )

    all_game_results = []
    init_func = partial(init_worker, STOCKFISH_PATH)

    with Pool(processes=num_workers, initializer=init_func) as pool:
        try:
            iterable = pool.imap_unordered(analyze_game, pgn_files)
            for result in tqdm(iterable, total=len(pgn_files), desc="Analyzing Games"):
                if result is not None:
                    all_game_results.append(result)
        finally:
            pool.terminate()
            pool.join()

    return all_game_results


def aggregate_results(all_game_results: list[dict]) -> tuple[defaultdict, defaultdict]:
    """Aggregates statistics from all analyzed games."""
    report = defaultdict(lambda: defaultdict(float))
    opening_report = defaultdict(lambda: defaultdict(int))

    for game_result in all_game_results:
        for stats in [game_result["white"], game_result["black"]]:
            player_name = "New Model" if "New" in stats.player_name else "Old Model"

            report[player_name]["games_played"] += 1
            report[player_name]["total_blunders"] += stats.blunders
            report[player_name]["total_mistakes"] += stats.mistakes
            report[player_name]["grand_total_cp_loss"] += stats.total_cp_loss
            report[player_name]["grand_total_moves"] += stats.move_count

            is_win = (stats.color == "White" and stats.final_result == "1-0") or (
                stats.color == "Black" and stats.final_result == "0-1"
            )

            if stats.was_winning:
                report[player_name]["winning_positions"] += 1
                if not is_win:
                    report[player_name]["throws"] += 1
            if stats.was_losing:
                report[player_name]["losing_positions"] += 1
                if is_win:
                    report[player_name]["comebacks"] += 1

        white_stats = game_result["white"]
        if white_stats.final_result == "1-0":
            player_name = (
                "New Model" if "New" in white_stats.player_name else "Old Model"
            )
            opening_report[white_stats.opening][player_name] += 1
        elif white_stats.final_result == "0-1":
            black_stats = game_result["black"]
            player_name = (
                "New Model" if "New" in black_stats.player_name else "Old Model"
            )
            opening_report[black_stats.opening][player_name] += 1
        else:
            opening_report[white_stats.opening]["Draws"] += 1

    return report, opening_report


def print_summary(report: defaultdict) -> dict:
    """Prints the final report card to the console and returns a log dictionary."""
    print("\n--- Chess Metrics Report Card ---")
    summary_log = {}
    for player in sorted(report.keys()):
        stats = report[player]
        games = stats["games_played"]
        moves = stats["grand_total_moves"]

        avg_blunders = stats["total_blunders"] / games if games > 0 else 0
        avg_mistakes = stats["total_mistakes"] / games if games > 0 else 0
        avg_acpl = stats["grand_total_cp_loss"] / moves if moves > 0 else 0
        throw_rate = (
            (stats["throws"] / stats["winning_positions"]) * 100
            if stats["winning_positions"] > 0
            else 0
        )
        comeback_rate = (
            (stats["comebacks"] / stats["losing_positions"]) * 100
            if stats["losing_positions"] > 0
            else 0
        )

        print(f"\n=== Player: {player} ({int(games)} games) ===")
        print(f"  Average Blunders per Game: {avg_blunders:.2f}")
        print(f"  Average Mistakes per Game: {avg_mistakes:.2f}")
        print(f"  Overall Average Centipawn Loss (ACPL): {avg_acpl:.2f}")
        print(
            f"  Position 'Throw' Rate: {throw_rate:.1f}% ({int(stats['throws'])}/{int(stats['winning_positions'])} winning positions)"
        )
        print(
            f"  Position 'Comeback' Rate: {comeback_rate:.1f}% ({int(stats['comebacks'])}/{int(stats['losing_positions'])} losing positions)"
        )

        prefix = "analysis"
        summary_log[f"{prefix}/{player}_ACPL"] = avg_acpl
        summary_log[f"{prefix}/{player}_Avg_Blunders"] = avg_blunders
        summary_log[f"{prefix}/{player}_Avg_Mistakes"] = avg_mistakes
        summary_log[f"{prefix}/{player}_Throw_Rate"] = throw_rate
        summary_log[f"{prefix}/{player}_Comeback_Rate"] = comeback_rate
    return summary_log


def log_to_wandb(summary_log: dict, opening_report: defaultdict, gen_id: str) -> None:
    """Logs the aggregated report and opening table to Weights & Biases."""
    print("\nLogging analysis report to Weights & Biases...")
    try:
        import wandb

        run = wandb.init(
            project=PROJECT_NAME,
            name=f"{gen_id}-stockfish-analysis",
            job_type="analysis",
            reinit=True,
        )
        opening_table = wandb.Table(
            columns=["Opening", "New Model Wins", "Old Model Wins", "Draws"]
        )
        for name, data in opening_report.items():
            opening_table.add_data(
                name,
                data.get("New Model", 0),
                data.get("Old Model", 0),
                data.get("Draws", 0),
            )

        summary_log["analysis/opening_performance"] = opening_table
        wandb.log(summary_log)
        print("Successfully logged analysis to W&B.")

    except Exception as e:
        print(f"Could not log to W&B. Error: {e}")
    finally:
        if "wandb" in locals() and wandb.run:
            wandb.finish()


# --- Main Execution ---


def main(game_dir: str, num_workers: int, gen_id: str) -> None:
    """Main function to run the analysis pipeline."""
    pgn_files = [
        os.path.join(game_dir, f) for f in os.listdir(game_dir) if f.endswith(".pgn")
    ]
    if not pgn_files:
        print("No .pgn files found in the directory.")
        return

    # 1. Run analysis
    all_game_results = run_analysis_pool(pgn_files, num_workers)
    if not all_game_results:
        print("No games were successfully analyzed.")
        return

    # 2. Aggregate results
    report, opening_report = aggregate_results(all_game_results)

    # 3. Print summary and get log data
    summary_log = print_summary(report)

    # 4. Log to external service
    log_to_wandb(summary_log, opening_report, gen_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze a directory of PGN games with Stockfish."
    )
    parser.add_argument(
        "--game-dir",
        type=str,
        default=EVAL_GAMES_DIR,
        help="Directory containing .pgn files to analyze.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of parallel analysis workers.",
    )
    parser.add_argument(
        "--gen-id", type=str, default="manual", help="Generation ID for this run."
    )
    args = parser.parse_args()

    main(args.game_dir, args.num_workers, args.gen_id)
