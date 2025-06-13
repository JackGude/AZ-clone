# state_encoder.py

import torch
import chess
from typing import Deque, List
from collections import deque

# This dictionary maps a piece type from the 'chess' library to a specific
# plane index (0-5). This is used for creating the input tensor.
PIECE_TO_PLANE = {
    chess.PAWN:   0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK:   3,
    chess.QUEEN:  4,
    chess.KING:   5,
}

def encode_history(history: Deque[chess.Board]) -> torch.FloatTensor:
    """
    Builds a (119, 8, 8) tensor representation of the last `history_size` moves.

    The 119 planes are structured as follows:
    - Frames 1 to 8 (T=8): Each frame represents a past board state.
      - 12 planes for piece positions (6 piece types * 2 colors).
      - 2 planes indicating if the position is a 2-fold or 3-fold repetition.
      Total historical planes: 8 frames * 14 planes/frame = 112 planes.

    - 7 additional planes representing constants for the CURRENT position:
      - 1 plane: The current side to move (1 for White, 0 for Black).
      - 1 plane: The total move number.
      - 4 planes: Castling rights for both players (WK, WQ, BK, BQ).
      - 1 plane: The half-move clock (for the 50-move rule).
    Total planes = 112 + 7 = 119.
    """
    T = 8
    assert len(history) <= T, "History deque may be shorter than T, but never longer"
    
    # Calculate the total number of channels (planes) in the final tensor.
    C = T * 14 + 7  # 14 planes per historical frame + 7 extra global planes
    x = torch.zeros(C, 8, 8, dtype=torch.float32)

    # --- 1) Stack the last T frames ---
    # Pad the history with `None` if the game has fewer than T moves.
    for i, fen_string in enumerate(history):
        b = chess.Board(fen_string)
        base = i * 14 # Calculate the starting plane index for this frame
        if b is None:
            continue # Skip empty (padded) frames

        # a) Piece planes (12 planes per frame)
        # For each piece on the board, set the corresponding cell to 1.0 in the correct plane.
        for sq, pc in b.piece_map().items():
            # White pieces are in planes 0-5, Black pieces are in planes 6-11.
            plane = PIECE_TO_PLANE[pc.piece_type] + (0 if pc.color == chess.WHITE else 6)
            r, c = divmod(sq, 8) # Convert square index (0-63) to (rank, file)
            x[base + plane, r, c] = 1.0

        # b) Repetition planes (2 planes per frame)
        # These planes are filled entirely with 1s if a repetition is detected.
        is_2_rep = b.is_repetition(2)
        is_3_rep = b.is_repetition(3)
        if is_2_rep: x[base + 12].fill_(1.0)
        if is_3_rep: x[base + 13].fill_(1.0)

    # --- 2) Add the 7 extra planes for the CURRENT position ---
    current_fen = history[-1]
    curr = chess.Board(current_fen) # The current board is the last one in the history
    extra_base = T * 14

    # Plane 112: Side to move (1.0 for White, 0.0 for Black)
    x[extra_base + 0].fill_(1.0 if curr.turn == chess.WHITE else 0.0)

    # Plane 113: Total move count, normalized to be in a small range.
    x[extra_base + 1].fill_(curr.fullmove_number / 100.0)

    # Planes 114-117: Castling rights. Each right gets its own plane.
    if curr.has_kingside_castling_rights(chess.WHITE):  x[extra_base + 2].fill_(1.0)
    if curr.has_queenside_castling_rights(chess.WHITE): x[extra_base + 3].fill_(1.0)
    if curr.has_kingside_castling_rights(chess.BLACK):  x[extra_base + 4].fill_(1.0)
    if curr.has_queenside_castling_rights(chess.BLACK): x[extra_base + 5].fill_(1.0)

    # Plane 118: No-progress count (for 50-move rule), normalized.
    x[extra_base + 6].fill_(curr.halfmove_clock / 100.0)

    # Final sanity check on the tensor shape.
    assert x.shape == (119, 8, 8)
    return x