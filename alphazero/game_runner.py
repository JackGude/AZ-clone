from dataclasses import dataclass
import sys
from typing import List, Optional, Tuple

import chess
from chess import pgn
import torch

from .env import ChessEnv
from .mcts import MCTS
from .move_encoder import MoveEncoder
from .state_encoder import encode_history
from .utils import load_or_initialize_model


@dataclass
class GameConfig:
    """Configuration settings for playing a single game of chess."""

    white_model_path: str
    black_model_path: str
    time_limit: float
    c_puct: float
    use_dynamic_cpuct: bool
    use_dirichlet_noise: bool
    dirichlet_alpha: float
    dirichlet_epsilon: float
    use_temperature_sampling: bool
    temp_threshold: int
    use_adjudication: bool
    adjudication_start_move: int
    draw_adjudication_threshold: float
    draw_adjudication_patience: int
    resign_threshold: float
    selfplay_max_moves: int
    verbose: bool = False
    log_prefix: str = ""


def adjudicate_game(
    q_value: float,
    move_count: int,
    config: GameConfig,
    draw_streak: int,
) -> Tuple[Optional[float], int]:
    """
    Helper function to check if a game should be adjudicated as a draw.
    """
    # --- 1. Do nothing if we are within the adjudication grace period ---
    if move_count <= config.adjudication_start_move:
        return None, 0

    # --- 2. Update draw streak counter ---
    if abs(q_value) < config.draw_adjudication_threshold:
        draw_streak += 1
    else:
        draw_streak = 0

    # --- 3. Check for draw outcome ---
    outcome = None
    if draw_streak >= config.draw_adjudication_patience:
        outcome = 0.0  # Adjudicated Draw

    return outcome, draw_streak


def play_game(
    config: GameConfig,
    env: Optional[ChessEnv] = None,
    pgn_node: Optional[pgn.GameNode] = None,
) -> Tuple[List[dict], str, int]:
    """
    Play a single game of chess using the provided configuration.
    Accepts an optional ChessEnv object.
    """
    # Initialize environment and move encoder
    if env is None:
        env = ChessEnv()
    move_encoder = MoveEncoder()

    # Load white model
    white_model, _ = load_or_initialize_model(config.white_model_path)

    if config.verbose:
        print(f"{config.log_prefix} White model ready")

    # Load or reuse black model
    if config.white_model_path == config.black_model_path:
        black_model = white_model
    else:
        black_model, _ = load_or_initialize_model(config.black_model_path)
        if config.verbose:
            print(f"{config.log_prefix} Black model ready")

    # Create MCTS instances
    dirichlet = config.dirichlet_alpha if config.use_dirichlet_noise else 0
    # Get device from the model
    device = next(white_model.parameters()).device

    white_mcts = MCTS(
        net=white_model,
        encoder=move_encoder,
        time_limit=config.time_limit,
        c_puct=config.c_puct,
        use_dynamic_cpuct=config.use_dynamic_cpuct,
        device=device,
        dirichlet_alpha=dirichlet,
        dirichlet_epsilon=config.dirichlet_epsilon,
    )
    black_mcts = MCTS(
        net=black_model,
        encoder=move_encoder,
        time_limit=config.time_limit,
        c_puct=config.c_puct,
        use_dynamic_cpuct=config.use_dynamic_cpuct,
        device=device,
        dirichlet_alpha=dirichlet,
        dirichlet_epsilon=config.dirichlet_epsilon,
    )

    # Initialize game state
    game_records = []
    move_count = 0
    draw_streak = 0
    outcome = None
    outcome_type = "unknown"

    # Main game loop
    while not env.board.is_game_over(claim_draw=True):
        if move_count >= config.selfplay_max_moves:
            outcome = 0.0  # Using 0.0 instead of a penalty for draws now
            outcome_type = "draw_cap"
            break

        is_white_turn = env.board.turn == chess.WHITE
        current_mcts = white_mcts if is_white_turn else black_mcts

        # Run MCTS search
        root, _ = current_mcts.run(env)

        # Adjudication (if enabled)
        if config.use_adjudication:
            adjudicated_outcome, draw_streak = adjudicate_game(
                q_value=root.Q,
                move_count=move_count,
                config=config,
                draw_streak=draw_streak,
            )
            if adjudicated_outcome is not None:
                outcome = adjudicated_outcome
                outcome_type = "draw_adjudicated"
                break

        # Resignation Check (FIXED)
        if root.Q < -config.resign_threshold:
            outcome = (
                -1.0 if is_white_turn else 1.0
            )  # Current player resigns, they lose
            outcome_type = "resign_white" if is_white_turn else "resign_black"
            break

        # Record game state for training if this is self-play
        if config.white_model_path == config.black_model_path:
            # Construct policy vector (pi) from MCTS visit counts (FIXED)
            counts = torch.zeros(move_encoder.mapping_size, dtype=torch.float32)
            for mv, child in root.children.items():
                counts[move_encoder.encode(mv)] = child.N

            if counts.sum() > 0:
                pi = counts / counts.sum()
                legality_mask = torch.zeros(
                    move_encoder.mapping_size, dtype=torch.float32
                )
                for move in env.board.legal_moves:
                    legality_mask[move_encoder.encode(move)] = 1.0

                game_records.append(
                    {
                        "state_history": list(env.history),
                        "pi": pi.numpy(),
                        "turn": env.board.turn,
                        "legality_mask": legality_mask.numpy(),  # Add the new data
                    }
                )

        # Select move based on temperature
        if config.use_temperature_sampling and move_count < config.temp_threshold:
            # Sample from policy distribution
            counts = torch.tensor(
                [child.N for child in root.children.values()], dtype=torch.float32
            )
            if counts.sum() > 0:
                policy_dist = counts / counts.sum()
                move_idx = torch.multinomial(policy_dist, 1).item()
                move = list(root.children.keys())[move_idx]
            else:  # Fallback if no moves explored
                move = list(env.board.legal_moves)[0]
        else:
            # Select move with highest visit count
            if not root.children:  # Fallback if no moves explored
                move = list(env.board.legal_moves)[0]
            else:
                move = max(root.children.items(), key=lambda item: item[1].N)[0]

        if config.verbose:
            try:
                move_san = env.board.san(move)
                q_value = root.Q
                player_turn_str = "White" if env.board.turn == chess.WHITE else "Black"
                print(
                    f"{config.log_prefix}    {env.board.fullmove_number}. {player_turn_str}: {move_san:<6} (Eval: {q_value:+.3f})",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as e:
                # Fallback in case SAN conversion fails
                print(
                    f"{config.log_prefix}    Playing move: {move}, Error in SAN: {e}",
                    file=sys.stderr,
                    flush=True,
                )

        if pgn_node:
            pgn_node = pgn_node.add_main_variation(move)

        # Make the move
        env.step(move)
        move_count += 1

    # --- Game is over, determine final outcome ---
    if outcome is None:
        board_outcome = env.board.outcome(claim_draw=True)
        if board_outcome is None:
            outcome = 0.0
            outcome_type = "draw_cap"
        else:
            # Determine the game outcome from white's perspective
            # outcome: 1.0 if white wins, -1.0 if black wins, 0.0 for draw
            if board_outcome.winner is True:  # White wins
                outcome = 1.0
                outcome_type = "checkmate_white"
            elif board_outcome.winner is False:  # Black wins
                outcome = -1.0
                outcome_type = "checkmate_black"
            else:  # Draw
                outcome = 0.0
                outcome_type = "draw_game"
    
    # Set final z-value for training examples
    final_examples = []
    if game_records:
        for record in game_records:
            # Calculate z_value from the perspective of the player to move
            # This flips the outcome for black's perspective
            z_value = outcome if record["turn"] == chess.WHITE else -outcome
            state_tensor = encode_history(record["state_history"])
            # Maintain the dictionary structure with all necessary fields
            final_examples.append(
                {
                    "state_history": state_tensor.numpy(),
                    "pi": record["pi"],
                    "z_value": z_value,
                    "turn": record["turn"],
                    "legality_mask": record["legality_mask"],
                }
            )

    return final_examples, outcome_type, move_count
