# evaluate.py

import torch
import chess
import argparse
import time
import json
import wandb
import csv
import sys
from automate import format_time

from config import (
    # Project and File Paths
    PROJECT_NAME,
    BEST_MODEL_PATH,
    CANDIDATE_MODEL_PATH,
    # Evaluation Config
    DEFAULT_NUM_EVAL_GAMES,
    DEFAULT_EVAL_TIME_LIMIT,
    DEFAULT_EVAL_CPUCT,
    DRAW_ADJUDICATION_THRESHOLD,
    DRAW_ADJUDICATION_PATIENCE,
    WIN_ADJUDICATION_THRESHOLD,
    WIN_ADJUDICATION_PATIENCE,
)
from alphazero.env import ChessEnv
from alphazero.model import AlphaZeroNet
from alphazero.mcts import MCTS
from alphazero.move_encoder import MoveEncoder

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Load openings from CSV
OPENING_BOOK_FENS = []
with open("openings/evaluation_openings.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        OPENING_BOOK_FENS.append((row["name"], row["fen"]))

# ─────────────────────────────────────────────────────────────────────────────
#  Game Playing Logic
# ─────────────────────────────────────────────────────────────────────────────


def play_match(net_white, net_black, encoder, time_limit, device, opening_info=None):
    """
    Plays a single game of chess between two neural networks.

    Args:
        net_white (nn.Module): The network playing as White.
        net_black (nn.Module): The network playing as Black.
        encoder (MoveEncoder): The move encoder/decoder utility.
        time_limit (int): The time in seconds for each MCTS search.
        device (str): The device to run inference on ('cpu' or 'cuda').

    Returns:
        float: The outcome of the game. +1.0 if White wins, -1.0 if Black wins, 0.0 for a draw.
    """
    env = ChessEnv()
    opening_name, opening_fen = opening_info
    env.set_board_from_fen(opening_fen)
    print(f"Starting game with opening: {opening_name}", file=sys.stderr, flush=True)

    # Create separate MCTS instances for each player.
    # Dirichlet noise is disabled (dirichlet_alpha=0) for deterministic evaluation.
    mcts_white = MCTS(
        net_white,
        encoder,
        time_limit=time_limit,
        c_puct=DEFAULT_EVAL_CPUCT,
        device=device,
        dirichlet_alpha=0,
    )
    mcts_black = MCTS(
        net_black,
        encoder,
        time_limit=time_limit,
        c_puct=DEFAULT_EVAL_CPUCT,
        device=device,
        dirichlet_alpha=0,
    )

    white_win_adjudication_streak = 0
    black_win_adjudication_streak = 0
    draw_adjudication_streak = 0

    # Main game loop
    while not env.board.is_game_over():
        is_white_turn = env.board.turn == chess.WHITE
        mcts = mcts_white if is_white_turn else mcts_black

        # Run the MCTS search to get the root node with visit counts
        root, _ = mcts.run(env)

        white_win_adjudication_streak, black_win_adjudication_streak, draw_adjudication_streak, outcome = adjudicate_game(
            q_value=root.Q,
            is_white_turn=is_white_turn,
            white_win_streak=white_win_adjudication_streak,
            black_win_streak=black_win_adjudication_streak,
            draw_streak=draw_adjudication_streak,
        )

        # If the game was adjudicated, outcome will be 1.0 or -1.0. End the match.
        if outcome is not None:
            return outcome

        # In evaluation, we deterministically choose the move with the highest visit count.
        counts = torch.zeros(encoder.mapping_size, dtype=torch.float32)
        for move, child in root.children.items():
            counts[encoder.encode(move)] = child.N

        if counts.sum() == 0:
            return 0.0

        move = encoder.decode(int(counts.argmax().item()))

        # --- Progress Tracking Line ---
        try:
            # Get the move in Standard Algebraic Notation (e.g., "Nf3") for easy reading
            move_san = env.board.san(move)
            # The Q-value from the root node represents the model's confidence from this position
            q_value = root.Q
            player_turn_str = "White" if env.board.turn == chess.WHITE else "Black"
            # Print a formatted status update
            print(
                f"    {env.board.fullmove_number}. {player_turn_str}: {move_san:<6} (Eval: {q_value:+.3f})",
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            # Fallback in case SAN conversion fails for some reason
            print(f"    Playing move: {move}", file=sys.stderr, flush=True)

        _, reward, done = env.step(move)
        if done:
            return reward

    # This should not be reached if the loop condition is correct, but as a fallback:
    return 0.0


def adjudicate_game(q_value, is_white_turn, white_win_streak, black_win_streak, draw_streak):
    """
    Checks if the game should be adjudicated based on a stable, decisive evaluation.

    Args:
        q_value (float): The MCTS Q-value from the current player's perspective.
        is_white_turn (bool): True if it is currently White's turn.
        white_win_streak (int): The current number of consecutive moves White has held an advantage.
        black_win_streak (int): The current number of consecutive moves Black has held an advantage.
        draw_streak (int): The current number of consecutive moves in the draw zone.

    Returns:
        tuple: A tuple containing three elements:
            (
                new_white_streak (int),
                new_black_streak (int),
                outcome (float or None): 1.0 for White win, -1.0 for Black win, or None if no adjudication.
            )
    """
    # Normalize the Q-value to always be from White's perspective
    q_white_perspective = q_value if is_white_turn else -q_value

    # --- 1. Update all streak counters based on the current evaluation ---
    
    # Update win streak
    if q_white_perspective > WIN_ADJUDICATION_THRESHOLD:
        white_win_streak += 1
    else:
        white_win_streak = 0

    # Update loss streak
    if q_white_perspective < -WIN_ADJUDICATION_THRESHOLD:
        black_win_streak += 1
    else:
        black_win_streak = 0
        
    # Update draw streak
    if abs(q_white_perspective) < DRAW_ADJUDICATION_THRESHOLD:
        draw_streak += 1
    else:
        draw_streak = 0

    # --- 2. Check for an outcome based on the updated streaks ---

    if white_win_streak >= WIN_ADJUDICATION_PATIENCE:
        print(f"  Game adjudicated as a WIN for WHITE...", file=sys.stderr, flush=True)
        return white_win_streak, black_win_streak, draw_streak, 1.0

    if black_win_streak >= WIN_ADJUDICATION_PATIENCE:
        print(f"  Game adjudicated as a WIN for BLACK...", file=sys.stderr, flush=True)
        return white_win_streak, black_win_streak, draw_streak, -1.0
        
    if draw_streak >= DRAW_ADJUDICATION_PATIENCE:
        print(f"  Game adjudicated as a DRAW...", file=sys.stderr, flush=True)
        return white_win_streak, black_win_streak, draw_streak, 0.0

    # --- 3. If no adjudication, return the updated streaks and no outcome ---
    return white_win_streak, black_win_streak, draw_streak, None

# ─────────────────────────────────────────────────────────────────────────────
#  Main Evaluation Function
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(
    checkpoint_old, checkpoint_new, num_games, time_limit, device, result_file
):
    """
    Orchestrates a head-to-head match between two network checkpoints.

    Args:
        checkpoint_old: Path to the old (incumbent) model checkpoint
        checkpoint_new: Path to the new (challenger) model checkpoint
        num_games: Number of games to play
        time_limit: Time limit per move in seconds
        device: Device to run inference on ('cpu' or 'cuda')
        result_file: Path to write the evaluation result JSON file
    """
    # Load the "old" (current best) network
    net_old = AlphaZeroNet(in_channels=119).to(device)
    net_old.load_state_dict(
        torch.load(checkpoint_old, map_location=device, weights_only=True)
    )
    net_old.eval()

    # Load the "new" (candidate) network
    net_new = AlphaZeroNet(in_channels=119).to(device)
    net_new.load_state_dict(
        torch.load(checkpoint_new, map_location=device, weights_only=True)
    )
    net_new.eval()

    encoder = MoveEncoder()

    total_start_time = time.time()

    # This list will store results from the perspective of the NEW network.
    # +1.0 = new_net won, -1.0 = new_net lost, 0.0 = draw.
    results = []

    print(
        f"\n--- Starting match: {num_games} games, {time_limit}s per move ---",
        file=sys.stderr,
        flush=True,
    )

    # Play the specified number of games, alternating colors
    for i in range(num_games):
        game_start_time = time.time()
        opening_info = OPENING_BOOK_FENS[i % len(OPENING_BOOK_FENS)]

        # New network plays as White in even-numbered games
        if i % 2 == 0:
            print(
                f"  Game {i+1}/{num_games}... (New plays as White)",
                file=sys.stderr,
                flush=True,
            )
            reward = play_match(
                net_new, net_old, encoder, time_limit, device, opening_info=opening_info
            )
            results.append(reward)
        # New network plays as Black in odd-numbered games
        else:
            print(
                f"  Game {i+1}/{num_games}... (New plays as Black)",
                file=sys.stderr,
                flush=True,
            )
            reward = play_match(
                net_old, net_new, encoder, time_limit, device, opening_info=opening_info
            )
            # The reward is from White's perspective. We negate it to keep
            # the score relative to the new network.
            results.append(-reward)

        # Print the outcome of the game using the last result in the list
        last_result = results[-1]
        if last_result == 1.0:
            outcome = "New model won"
        elif last_result == -1.0:
            outcome = "Old model won"
        else:
            outcome = "Draw"
        print(
            f"  Game {i+1} finished. Outcome: {outcome}. Current score (New vs Old): {sum(r for r in results if r==1)} - {sum(1 for r in results if r==-1)}",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"Time: {format_time(time.time() - game_start_time)}",
            file=sys.stderr,
            flush=True,
        )

    total_eval_time = time.time() - total_start_time
    # Calculate final statistics from the perspective of the new network
    wins = sum(1 for r in results if r == 1.0)
    draws = sum(1 for r in results if r == 0.0)
    losses = sum(1 for r in results if r == -1.0)
    # Win rate is calculated as wins + half of the draws
    win_rate = (wins + 0.5 * draws) / num_games if num_games > 0 else 0.0
    avg_game_time = total_eval_time / num_games if num_games > 0 else 0.0

    print("\n=== Evaluation Summary ===", file=sys.stderr, flush=True)
    print(
        f"Results for NEW model vs OLD model: {wins} Wins, {losses} Losses, {draws} Draws",
        file=sys.stderr,
        flush=True,
    )
    print(f"Win Rate of NEW model: {win_rate*100:.1f}%", file=sys.stderr, flush=True)

    # Log the detailed evaluation results to wandb
    if wandb.run:
        wandb.log(
            {
                "eval_num_games": num_games,
                "avg_game_time": avg_game_time,
                "total_eval_time": total_eval_time,
                "eval_win_rate": win_rate,
                "eval_wins": wins,
                "eval_draws": draws,
                "eval_losses": losses,
            }
        )

    # Write the win rate to a JSON file
    result = {"win_rate": win_rate}
    with open(result_file, "w") as f:
        json.dump(result, f)


def main(args):
    """Main function for standalone execution."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize wandb with the new format
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
        device=device,
        result_file=args.result_file,
    )

    if wandb.run:
        wandb.run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate two AlphaZeroNet checkpoints head-to-head."
    )
    parser.add_argument(
        "--old",
        type=str,
        default=BEST_MODEL_PATH,
        help="Path to the 'old' incumbent best checkpoint, default is the best model in the checkpoints directory",
    )
    parser.add_argument(
        "--new",
        type=str,
        default=CANDIDATE_MODEL_PATH,
        help="Path to the 'new' challenger checkpoint, default is the candidate model in the checkpoints directory",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=DEFAULT_NUM_EVAL_GAMES,
        help="Number of games to play, default is 24",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=DEFAULT_EVAL_TIME_LIMIT,
        help="Time limit in seconds per move, default is 10",
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
        help="Path to write the evaluation result JSON file, default is logs/eval_results.json",
    )
    args = parser.parse_args()

    main(args)
