# move_encoder.py
import torch
import chess
import numpy as np

class MoveEncoder:
    """
    Handles the encoding and decoding of chess moves into a fixed-size vector,
    based on the 8x8x73 move representation scheme from AlphaZero.
    
    This creates a total of 4672 (64 squares * 73 move types) possible move indices.
    """
    def __init__(self):
        # Define the basic move types used to build the full mapping.
        
        # Directions for sliding pieces (rooks, bishops, queens). 8 directions.
        self.directions = [
            ( 1,  0), (-1,  0), ( 0,  1), ( 0, -1), # Rook directions
            ( 1,  1), ( 1, -1), (-1,  1), (-1, -1), # Bishop directions
        ]
        # Offsets for knight moves. 8 possible hops.
        self.knight_offsets = [
            (-2, -1), (-2,  1), (-1, -2), (-1,  2),
            ( 1, -2), ( 1,  2), ( 2, -1), ( 2,  1),
        ]
        # Piece types for pawn under-promotions (excluding the queen).
        self.promo_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

        # Build the master mapping list. This list will have 4672 entries.
        # Each entry is a tuple: (from_square, to_square, promotion_piece)
        self.mapping = []
        for from_sq in range(64):
            r0, f0 = divmod(from_sq, 8) # Get rank and file of the starting square

            # 1) Queen-like moves (56 slots):
            #    These represent sliding moves. Queen promotions are also encoded here.
            #    8 directions * 7 max distance = 56 slots.
            for dr, df in self.directions:
                for dist in range(1, 8):
                    r, f = r0 + dr*dist, f0 + df*dist
                    if 0 <= r < 8 and 0 <= f < 8:
                        # This is a legal move on the board.
                        self.mapping.append((from_sq, r*8 + f, None))
                    else:
                        # This move is off-board; add a placeholder.
                        self.mapping.append((from_sq, None, None))

            # 2) Knight moves (8 slots):
            for dr, df in self.knight_offsets:
                r, f = r0 + dr, f0 + df
                if 0 <= r < 8 and 0 <= f < 8:
                    self.mapping.append((from_sq, r*8 + f, None))
                else:
                    self.mapping.append((from_sq, None, None))

            # 3) Pawn under-promotions (9 slots):
            #    3 directions (straight, capture left, capture right) * 3 pieces (N, B, R) = 9 slots.
            #    The logic handles both White and Black promotions, folding them into the same slots.
            for df in (0, -1, 1): # Move direction: straight, left, right
                # Check for White promotions (rank 6 to 7)
                if r0 == 6 and 0 <= f0 + df < 8:
                    to_sq = (r0+1)*8 + (f0+df)
                    for pc in self.promo_pieces:
                        self.mapping.append((from_sq, to_sq, pc))
                # Check for Black promotions (rank 1 to 0)
                elif r0 == 1 and 0 <= f0 + df < 8:
                    to_sq = (r0-1)*8 + (f0+df)
                    for pc in self.promo_pieces:
                        self.mapping.append((from_sq, to_sq, pc))
                else:
                    # If not a promotion rank, add placeholders for all 3 pieces.
                    for _ in self.promo_pieces:
                        self.mapping.append((from_sq, None, None))

        # Sanity check to ensure the mapping has the correct size.
        assert len(self.mapping) == 64 * 73, f"Built {len(self.mapping)}, expected 4672"

        # Build a reverse lookup dictionary for fast encoding.
        # It maps the tuple (from, to, promo) to its index.
        self._index = {triple: idx for idx, triple in enumerate(self.mapping)}
        self.mapping_size = len(self.mapping)
        self._build_flip_map()

    def _build_flip_map(self):
        """
        Creates a map to translate policy indices when the board is flipped.
        This is used for data augmentation.
        """
        self.flip_map = np.arange(self.mapping_size)
        for idx, (f, t, p) in enumerate(self.mapping):
            if f is not None and t is not None:
                # chess.square_mirror() correctly flips the square index horizontally
                flipped_from = chess.square_mirror(f)
                flipped_to = chess.square_mirror(t)
                flipped_key = (flipped_from, flipped_to, p)
                if flipped_key in self._index:
                    self.flip_map[idx] = self._index[flipped_key]

    def encode(self, move: chess.Move) -> int:
        """
        Maps a chess.Move object to its corresponding integer index (0-4671).
        
        Note on queen promotions: A promotion to a queen (move.promotion == chess.QUEEN)
        is treated as a normal sliding move (promo_key = None), so it gets encoded in
        the first 56 "queen-like move" slots. This is a key part of the AlphaZero design.
        """
        # A move's promotion piece is only relevant if it's an under-promotion.
        p = move.promotion if move.promotion in self.promo_pieces else None
        
        # Create the key tuple to look up in our index.
        key = (move.from_square, move.to_square, p)
        
        if key not in self._index:
            raise ValueError(f"Move {move.uci()} not found in the 8x8x73 mapping.")
        
        return self._index[key]

    def decode(self, idx: int) -> chess.Move:
        """Maps an integer index back to a chess.Move object."""
        # Retrieve the (from, to, promo) tuple from the master list.
        f, t, p = self.mapping[idx]
        if f is None or t is None:
            raise ValueError(f"Index {idx} corresponds to a placeholder (illegal) move.")
        
        return chess.Move(from_square=f, to_square=t, promotion=p)