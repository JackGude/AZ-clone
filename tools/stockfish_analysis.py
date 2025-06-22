# stockfish_analysis.py
import chess
import chess.pgn
import chess.engine
import argparse
import os
import wandb
from multiprocessing import Pool
from collections import defaultdict

# --- Local Project Imports ---
# Assuming you might run this via automate.py eventually
from config import STOCKFISH_PATH, EVAL_GAMES_DIR, NUM_WORKERS, PROJECT_NAME


def analyze_game(pgn_path: str) -> dict:
    """
    Analyzes a single PGN file with Stockfish to calculate metrics FOR BOTH PLAYERS.
    """
    try:
        with open(pgn_path, "r") as f:
            game = chess.pgn.read_game(f)
        if game is None:
            return None
    except Exception as e:
        print(f"Error reading PGN {pgn_path}: {e}")
        return None

    # --- Setup dictionaries for both players ---
    white_player_name = game.headers.get("White", "Unknown")
    black_player_name = game.headers.get("Black", "Unknown")

    white_results = {
        "player": white_player_name,
        "blunders": 0,
        "mistakes": 0,
        "total_cp_loss": 0,
        "was_winning": False,
        "was_losing": False,
    }
    black_results = {
        "player": black_player_name,
        "blunders": 0,
        "mistakes": 0,
        "total_cp_loss": 0,
        "was_winning": False,
        "was_losing": False,
    }

    board = game.board()
    move_count = 0

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        for node in game.mainline():
            is_whites_move = board.turn == chess.WHITE

            info = engine.analyse(board, chess.engine.Limit(time=0.5))
            score_before = info["score"].white().score(mate_score=10000)

            board.push(node.move)
            info_after = engine.analyse(board, chess.engine.Limit(time=0.1))
            score_after = info_after["score"].white().score(mate_score=10000)

            cp_loss = score_before - score_after

            # Attribute the loss to the player who just moved
            if is_whites_move:
                white_results["total_cp_loss"] += cp_loss
                if cp_loss > 100:
                    white_results["blunders"] += 1
                elif cp_loss > 50:
                    white_results["mistakes"] += 1
            else:
                # Black's cp_loss is the negative of White's change
                black_cp_loss = -cp_loss
                black_results["total_cp_loss"] += black_cp_loss
                if black_cp_loss > 100:
                    black_results["blunders"] += 1
                elif black_cp_loss > 50:
                    black_results["mistakes"] += 1

            # Track if either player had a winning advantage from White's perspective
            if score_before > 200:
                white_results["was_winning"] = True
            if score_before < -200:
                black_results["was_winning"] = (
                    True  # Black is winning if score is <-2.0
                )

            move_count += 1

    # Add final game info
    for res in [white_results, black_results]:
        res["opening"] = game.headers.get("Opening", "Unknown")
        res["final_result"] = game.headers.get("Result", "*")
        res["color"] = "White" if res["player"] == white_player_name else "Black"

    return {"white": white_results, "black": black_results, "moves": move_count}


def generate_report_card(game_dir: str, num_workers: int, gen_id: str):
    """
    Analyzes all PGNs in a directory and prints a comprehensive, fair report card.
    """
    pgn_files = [
        os.path.join(game_dir, f) for f in os.listdir(game_dir) if f.endswith(".pgn")
    ]
    if not pgn_files:
        print("No .pgn files found in the directory.")
        return

    print(
        f"Found {len(pgn_files)} games to analyze. Starting analysis with {num_workers} workers..."
    )

    with Pool(processes=num_workers) as pool:
        results = pool.map(analyze_game, pgn_files)

    # --- Aggregate results ---
    report = defaultdict(lambda: defaultdict(float))
    opening_report = defaultdict(lambda: defaultdict(int))

    for game_result in results:
        if game_result is None:
            continue

        white_stats = game_result["white"]
        black_stats = game_result["black"]

        # Process stats for both players
        for stats in [white_stats, black_stats]:
            player_name = "New Model" if "New" in stats["player"] else "Old Model"

            report[player_name]["games_played"] += 1
            report[player_name]["total_blunders"] += stats["blunders"]
            report[player_name]["total_acpl"] += stats["acpl"]

            # Check for throws (was winning, but didn't win)
            is_win = (stats["color"] == "White" and stats["final_result"] == "1-0") or (
                stats["color"] == "Black" and stats["final_result"] == "0-1"
            )

            if stats["was_winning"] and not is_win:
                report[player_name]["throws"] += 1
            if stats["was_winning"]:
                report[player_name]["winning_positions"] += 1

        # Aggregate Opening Stats (this part can be simpler)
        opening = white_stats["opening"]
        if white_stats["final_result"] == "1-0":
            opening_report[opening][white_stats["player"]] += 1
        elif black_stats["final_result"] == "0-1":
            opening_report[opening][black_stats["player"]] += 1
        else:
            opening_report[opening]["Draws"] += 1

    # --- Print Report Card ---
    print("\n--- Chess Metrics Report Card ---")
    for player in ["New Model", "Old Model"]:
        stats = report[player]
        games = stats["games_played"]
        avg_blunders = stats["total_blunders"] / games if games > 0 else 0
        avg_acpl = stats["total_acpl"] / games if games > 0 else 0
        throw_rate = (
            (stats["throws"] / stats["winning_positions"]) * 100
            if stats["winning_positions"] > 0
            else 0
        )

        print(f"\n=== Player: {player} ({int(games)} games total) ===")
        print(f"  Average Blunders per Game: {avg_blunders:.2f}")
        print(f"  Average Centipawn Loss (ACPL): {avg_acpl:.2f}")
        print(
            f"  Position 'Throw' Rate: {throw_rate:.1f}% ({int(stats['throws'])}/{int(stats['winning_positions'])} winning positions)"
        )

    # --- Log to W&B ---
    print("\nLogging analysis report to Weights & Biases...")
    try:
        wandb.init(project=PROJECT_NAME, name=f"{gen_id}-stockfish-analysis", job_type="analysis", reinit=True)
        
        # 1. Log the high-level summary metrics for both players
        summary_log = {}
        for player in ["New Model", "Old Model"]:
            stats = report[player]
            games_played = stats.get('games_played', 0)
            winning_positions = stats.get('winning_positions', 0)
            
            # Use .get(key, 0) to avoid errors if a player had no games/stats
            avg_blunders = stats.get('total_blunders', 0) / games_played if games_played > 0 else 0
            avg_acpl = stats.get('total_acpl', 0) / games_played if games_played > 0 else 0
            throw_rate = (stats.get('throws', 0) / winning_positions) * 100 if winning_positions > 0 else 0
            
            # Use a prefix to group these metrics in the W&B dashboard
            prefix = "analysis"
            summary_log[f"{prefix}/{player}_ACPL"] = avg_acpl
            summary_log[f"{prefix}/{player}_Avg_Blunders"] = avg_blunders
            summary_log[f"{prefix}/{player}_Throw_Rate"] = throw_rate

        # 2. Create and log the detailed opening performance table
        opening_table = wandb.Table(columns=["Opening", "New Model Wins", "Old Model Wins", "Draws"])
        for name, data in opening_report.items():
            opening_table.add_data(
                name, 
                data.get("New Model", 0), 
                data.get("Old Model", 0), 
                data.get("Draws", 0)
            )
        
        summary_log["analysis/opening_performance"] = opening_table
        
        wandb.log(summary_log)
        print("Successfully logged analysis to W&B.")

    except Exception as e:
        print(f"Could not log to W&B. Error: {e}")
    finally:
        if wandb.run:
            wandb.finish()


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

    generate_report_card(args.game_dir, args.num_workers, args.gen_id)
