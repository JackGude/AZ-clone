# evaluate.py

import os
import torch
import chess
import argparse
import time
import json
import wandb
from datetime import datetime

from alphazero.env import ChessEnv
from alphazero.model import AlphaZeroNet
from alphazero.mcts import MCTS
from alphazero.move_encoder import MoveEncoder

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = "checkpoints"
# Default time limit for evaluation games if not provided via command line
DEFAULT_TIME_LIMIT = 5

# ─────────────────────────────────────────────────────────────────────────────
#  Game Playing Logic
# ─────────────────────────────────────────────────────────────────────────────

def play_match(net_white, net_black, encoder, time_limit, c_puct, device):
    """
    Plays a single game of chess between two neural networks.

    Args:
        net_white (nn.Module): The network playing as White.
        net_black (nn.Module): The network playing as Black.
        encoder (MoveEncoder): The move encoder/decoder utility.
        time_limit (int): The time in seconds for each MCTS search.
        c_puct (float): The exploration constant for MCTS.
        device (str): The device to run inference on ('cpu' or 'cuda').

    Returns:
        float: The outcome of the game. +1.0 if White wins, -1.0 if Black wins, 0.0 for a draw.
    """
    env = ChessEnv(history_size=8)
    env.reset()

    # Create separate MCTS instances for each player.
    # Dirichlet noise is disabled (dirichlet_alpha=0) for deterministic evaluation.
    mcts_white = MCTS(net_white, encoder, time_limit=time_limit, c_puct=c_puct, device=device, dirichlet_alpha=0)
    mcts_black = MCTS(net_black, encoder, time_limit=time_limit, c_puct=c_puct, device=device, dirichlet_alpha=0)

    # Main game loop
    while not env.board.is_game_over():
        is_white_turn = env.board.turn == chess.WHITE
        mcts = mcts_white if is_white_turn else mcts_black

        # Run the MCTS search to get the root node with visit counts
        root, _ = mcts.run(env)

        # In evaluation, we deterministically choose the move with the highest visit count.
        counts = torch.zeros(encoder.mapping_size, dtype=torch.float32)
        for move, child in root.children.items():
            counts[encoder.encode(move)] = child.N
        
        move = encoder.decode(int(counts.argmax().item()))

        _, reward, done = env.step(move)
        if done:
            return reward

    # This should not be reached if the loop condition is correct, but as a fallback:
    return 0.0

# ─────────────────────────────────────────────────────────────────────────────
#  Main Evaluation Function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    checkpoint_old,
    checkpoint_new,
    num_games,
    time_limit,
    c_puct,
    device
):
    """
    Orchestrates a head-to-head match between two network checkpoints.
    """
    # Load the "old" (current best) network
    net_old = AlphaZeroNet(in_channels=119).to(device)
    net_old.load_state_dict(torch.load(checkpoint_old, map_location=device, weights_only=True))
    net_old.eval()

    # Load the "new" (candidate) network
    net_new = AlphaZeroNet(in_channels=119).to(device)
    net_new.load_state_dict(torch.load(checkpoint_new, map_location=device, weights_only=True))
    net_new.eval()

    encoder = MoveEncoder()
    
    # This list will store results from the perspective of the NEW network.
    # +1.0 = new_net won, -1.0 = new_net lost, 0.0 = draw.
    results = []

    print(f"\n--- Starting match: {num_games} games, {time_limit}s per move ---")

    # Play the specified number of games, alternating colors
    for i in range(num_games):
        # New network plays as White in even-numbered games
        if i % 2 == 0:
            print(f"  Game {i+1}/{num_games}... (New plays White)")
            reward = play_match(net_new, net_old, encoder, time_limit, c_puct, device)
            results.append(reward)
        # New network plays as Black in odd-numbered games
        else:
            print(f"  Game {i+1}/{num_games}... (New plays Black)")
            reward = play_match(net_old, net_new, encoder, time_limit, c_puct, device)
            # The reward is from White's perspective. We negate it to keep
            # the score relative to the new network.
            results.append(-reward)
        
        print(f"  Game {i+1} finished. Current score (New vs Old): {sum(r for r in results if r==1)} - {sum(1 for r in results if r==-1)}")

    # Calculate final statistics from the perspective of the new network
    wins = sum(1 for r in results if r == 1.0)
    draws = sum(1 for r in results if r == 0.0)
    losses = sum(1 for r in results if r == -1.0)
    
    # Win rate is calculated as wins + half of the draws
    win_rate = (wins + 0.5 * draws) / num_games if num_games > 0 else 0.0

    print("\n=== Evaluation Summary ===")
    print(f"Results for NEW model vs OLD model: {wins} Wins, {losses} Losses, {draws} Draws")
    print(f"Win Rate of NEW model: {win_rate*100:.1f}%")
    
    # Log the detailed evaluation results to wandb
    if wandb.run:
        wandb.log({
            "eval_num_games": num_games,
            "eval_win_rate": win_rate,
            "eval_wins": wins,
            "eval_draws": draws,
            "eval_losses": losses,
        })

    # Print the final win rate in a machine-readable format for automate.py to capture
    print(f"FINAL_WIN_RATE: {win_rate}")


def main(args):
    """Main function for standalone execution."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize wandb based on --no-wandb flag
    wandb_mode = "disabled" if args.no_wandb else "online"
    wandb.init(project="alphazero-chess", resume="allow", mode=wandb_mode)

    evaluate(
        checkpoint_old=args.old,
        checkpoint_new=args.new,
        num_games=args.games,
        time_limit=args.time_limit,
        c_puct=args.cpuct,
        device=device
    )

    if wandb.run:
        wandb.run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate two AlphaZeroNet checkpoints head-to-head.")
    parser.add_argument("--old",   type=str, required=True, help="Path to the 'old' challenger checkpoint")
    parser.add_argument("--new",   type=str, required=True, help="Path to the 'new' incumbent checkpoint")
    parser.add_argument("--games", type=int, default=20, help="Number of games to play")
    parser.add_argument("--time-limit", type=int, default=DEFAULT_TIME_LIMIT, help="Time limit in seconds per move")
    parser.add_argument("--cpuct", type=float, default=1.0, help="PUCT constant for MCTS")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging.")
    args = parser.parse_args()
    
    main(args)