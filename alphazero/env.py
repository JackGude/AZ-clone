# alphazero/env.py

import chess
import numpy as np
from collections import deque

# We import the state encoder here as it's a core dependency of the environment.
from .state_encoder import encode_history

class ChessEnv:
    """
    A wrapper for the python-chess library to manage the game state,
    including the crucial history of board positions for the neural network input.
    """

    def __init__(self):
        """
        Initializes the chess environment.
        """
        # The history size is a fixed architectural constant of the neural network.
        # It is set to 8, representing the current position plus the last 7 half-moves,
        # which is standard for AlphaZero-style models.
        self.history_size = 8
        
        self.board = chess.Board()
        # The history is a deque (a double-ended queue) with a fixed maximum length.
        # When a new item is added and the deque is full, the oldest item is automatically discarded.
        self.history = deque(maxlen=self.history_size)
        
        # Initialize the board and history to the starting position.
        self.reset()

    def reset(self):
        """
        Resets the environment to the standard starting chess position and clears the history.
        """
        self.board.reset()
        self.history.clear()
        # Pre-fill the history buffer with the starting position's FEN.
        # This ensures the buffer is always the correct size for the neural network.
        for _ in range(self.history_size):
            self.history.append(self.board.fen())

    def step(self, move):
        """
        Applies a move to the board, updates the history, and returns the results.

        Args:
            move (chess.Move): The move to be played.

        Returns:
            tuple: A tuple containing (observation, reward, done), where 'done' is a boolean
                   indicating if the game is over.
        """
        self.board.push(move)
        # Append the new board state to our history deque.
        self.history.append(self.board.fen())
        
        done = self.board.is_game_over()
        reward = 0.0
        if done:
            outcome = self.board.outcome()
            if outcome.winner is True:  # White wins
                reward = 1.0
            elif outcome.winner is False: # Black wins
                reward = -1.0
            # Otherwise, reward remains 0.0 for a draw.
        
        return self.history, reward, done

    def legal_moves(self):
        """Returns a list of all legal moves from the current position."""
        return list(self.board.legal_moves)
        
    def get_state_tensor(self, board=None):
        """
        Generates the NN input tensor from a given board state. If no board is provided,
        it uses the environment's current board. This is the crucial method used by MCTS
        to evaluate hypothetical positions during its search.

        Args:
            board (chess.Board, optional): The board state to encode. Defaults to None.

        Returns:
            torch.Tensor: The encoded state tensor ready for the neural network.
        """
        if board is None:
            board = self.board

        # 1. Reconstruct the FEN history from the board's internal move stack.
        fen_history = []
        temp_board = board.copy()
        
        # We pop moves to go backward in time, collecting up to `history_size` states.
        for _ in range(self.history_size):
            fen_history.append(temp_board.fen())
            try:
                temp_board.pop()
            except IndexError:
                # Reached the beginning of the game before filling history.
                break
        
        # The history is built from newest to oldest, so it must be reversed.
        fen_history.reverse()

        # 2. Pad the beginning of the history with the earliest known position
        # if the game is shorter than the required history length.
        first_known_fen = fen_history[0]
        while len(fen_history) < self.history_size:
            fen_history.insert(0, first_known_fen)

        history_deque = deque(fen_history, maxlen=self.history_size)
    
        # 3. Get the 3D tensor from your encoder
        state_tensor_3d = encode_history(history_deque)
        
        # --- THIS IS THE FIX ---
        # 4. Add a "batch" dimension to the front, changing shape from (C, H, W)
        #    to (1, C, H, W) before returning.
        return state_tensor_3d.unsqueeze(0)

    def set_board_from_moves(self, move_string):
        """
        A helper function for opening books. Creates a board state by playing a sequence
        of moves, ensuring the history is correctly populated.
        """
        self.reset() # Start from a clean slate
        try:
            for move in move_string.split(' '):
                self.step(self.board.parse_san(move))
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse move string '{move_string}' in opening book. Error: {e}. Starting from standard position.")
            self.reset()

    def set_board_from_fen(self, fen):
        """
        Sets up the board from a FEN string and initializes the history accordingly.
        
        Args:
            fen (str): A valid FEN string representing the desired board position.
        """
        try:
            self.board = chess.Board(fen)
            self.history.clear()
            # Pre-fill the history buffer with the current position's FEN
            for _ in range(self.history_size):
                self.history.append(self.board.fen())
        except ValueError as e:
            print(f"Warning: Invalid FEN string '{fen}'. Error: {e}. Starting from standard position.")
            self.reset()