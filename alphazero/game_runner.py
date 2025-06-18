from dataclasses import dataclass
import torch
import chess
from typing import List, Tuple, Optional
import sys

from .env import ChessEnv
from .mcts import MCTS
from .model import AlphaZeroNet
from .state_encoder import encode_history
from .move_encoder import MoveEncoder


@dataclass
class GameConfig:
    """Configuration settings for playing a single game of chess."""

    white_model_path: str
    black_model_path: str
    time_limit: float
    c_puct: float
    use_dirichlet_noise: bool
    dirichlet_alpha: float
    dirichlet_epsilon: float
    use_temperature_sampling: bool
    temp_threshold: int
    use_adjudication: bool
    win_adjudication_threshold: float
    win_adjudication_patience: int
    draw_adjudication_threshold: float
    draw_adjudication_patience: int
    resign_threshold: float
    selfplay_max_moves: int
    verbose: bool = False
    log_prefix: str = ""


def adjudicate_game(
    q_value: float,
    is_white_turn: bool,
    win_threshold: float,
    win_patience: int,
    draw_threshold: float,
    draw_patience: int,
    white_win_streak: int,
    black_win_streak: int,
    draw_streak: int,
) -> Tuple[Optional[float], int, int, int]:
    """
    Helper function to check if a game should be adjudicated.
    Now takes the post-MCTS Q-value directly.
    """
    # --- 1. Normalize the Q-value to always be from White's perspective ---
    q_white_perspective = q_value if is_white_turn else -q_value

    # --- 2. Update all streak counters based on the current evaluation ---
    # Update win streak
    if q_white_perspective > win_threshold:
        white_win_streak += 1
    else:
        white_win_streak = 0
    # Update loss streak
    if q_white_perspective < -win_threshold:
        black_win_streak += 1
    else:
        black_win_streak = 0
    # Update draw streak
    if abs(q_white_perspective) < draw_threshold:
        draw_streak += 1
    else:
        draw_streak = 0

    # --- 3. Check for an outcome based on the updated streaks ---
    outcome = None
    if white_win_streak >= win_patience:
        outcome = 1.0  # White win
    elif black_win_streak >= win_patience:
        outcome = -1.0  # Black win
    elif draw_streak >= draw_patience:
        outcome = 0.0  # Draw

    return outcome, white_win_streak, black_win_streak, draw_streak


def play_game(
    config: GameConfig, env: Optional[ChessEnv] = None
) -> Tuple[List[dict], str, int]:
    """
    Play a single game of chess using the provided configuration.
    Accepts an optional ChessEnv object.
    """
    # Initialize environment and move encoder
    if env is None:
        env = ChessEnv()
    move_encoder = MoveEncoder()

    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    white_model = AlphaZeroNet().to(device)
    white_model.load_state_dict(
        torch.load(config.white_model_path, map_location=device, weights_only=True)
    )
    if config.verbose:
        print(f"{config.log_prefix} Loaded white model from {config.white_model_path}")
    white_model.eval()

    if config.white_model_path == config.black_model_path:
        black_model = white_model
    else:
        black_model = AlphaZeroNet().to(device)
        black_model.load_state_dict(
            torch.load(config.black_model_path, map_location=device, weights_only=True)
        )
        if config.verbose:
            print(f"{config.log_prefix} Loaded black model from {config.black_model_path}")
        black_model.eval()

    # Create MCTS instances
    dirichlet = config.dirichlet_alpha if config.use_dirichlet_noise else 0
    white_mcts = MCTS(
        net=white_model,
        encoder=move_encoder,
        time_limit=config.time_limit,
        c_puct=config.c_puct,
        device=device,
        dirichlet_alpha=dirichlet,
        dirichlet_epsilon=config.dirichlet_epsilon,
    )
    black_mcts = MCTS(
        net=black_model,
        encoder=move_encoder,
        time_limit=config.time_limit,
        c_puct=config.c_puct,
        device=device,
        dirichlet_alpha=dirichlet,
        dirichlet_epsilon=config.dirichlet_epsilon,
    )

    # Initialize game state
    game_records = []
    move_count = 0
    white_win_streak, black_win_streak, draw_streak = 0, 0, 0
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
            adjudicated_outcome, white_win_streak, black_win_streak, draw_streak = (
                adjudicate_game(
                    q_value=root.Q,
                    is_white_turn=is_white_turn,
                    win_threshold=config.win_adjudication_threshold,
                    win_patience=config.win_adjudication_patience,
                    draw_threshold=config.draw_adjudication_threshold,
                    draw_patience=config.draw_adjudication_patience,
                    white_win_streak=white_win_streak,
                    black_win_streak=black_win_streak,
                    draw_streak=draw_streak,
                )
            )
            if adjudicated_outcome is not None:
                outcome = adjudicated_outcome
                if outcome == 1.0:
                    outcome_type = "white_win_adjudicated"
                elif outcome == -1.0:
                    outcome_type = "black_win_adjudicated"
                else:
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
                game_records.append(
                    {
                        "state_history": list(env.history),
                        "pi": pi.numpy(),
                        "turn": env.board.turn,
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

        # Make the move
        env.step(move)
        move_count += 1

    # --- Game is over, determine final outcome ---
    if outcome is None:
        board_outcome = env.board.outcome(claim_draw=True)
        if board_outcome is None:  # Should only happen on move cap
            outcome = 0.0
            outcome_type = "draw_cap"
        else:
            outcome = (
                0.0
                if board_outcome.winner is None
                else 1.0
                if board_outcome.winner
                else -1.0
            )
            outcome_type = (
                "checkmate_white"
                if outcome == 1.0
                else "checkmate_black"
                if outcome == -1.0
                else "draw_game"
            )

    # Set final z-value for training examples
    final_examples = []
    if game_records:
        for record in game_records:
            z_value = outcome if record["turn"] == chess.WHITE else -outcome
            state_tensor = encode_history(record["state_history"])
            final_examples.append((state_tensor.numpy(), record["pi"], z_value))

    return final_examples, outcome_type, move_count
