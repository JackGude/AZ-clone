# env.py
import chess
from collections import deque

class ChessEnv:
    def __init__(self, history_size: int = 8):
        self.history_size = history_size
        self._reset_history()

    def _reset_history(self):
        self.history = deque(maxlen=self.history_size)

    def reset(self):
        """Start a new game and clear history."""
        self.board = chess.Board()
        self._reset_history()
        self.history.append(self.board.copy())
        return self.board

    def set_board(self, board_or_fen):
        """
        Set up the board with either a FEN string or a chess.Board object.
        
        Args:
            board_or_fen: Either a FEN string or a chess.Board object
        """
        if isinstance(board_or_fen, str):
            self.board = chess.Board(board_or_fen)
        elif isinstance(board_or_fen, chess.Board):
            self.board = board_or_fen.copy()
        else:
            raise ValueError("board_or_fen must be either a FEN string or a chess.Board object")
        
        self._reset_history()
        self.history.append(self.board.copy())
        return self.board

    def legal_moves(self):
        return list(self.board.legal_moves)

    def step(self, move):
        self.board.push(move)
        self.history.append(self.board.copy())

        done = self.board.is_game_over()
        if not done:
            return self.board, 0.0, False

        outcome = self.board.outcome()
        if outcome.winner is None:
            reward = 0.0
        else:
            reward = 1.0 if outcome.winner == chess.WHITE else -1.0
        return self.board, reward, True
