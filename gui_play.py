# gui_play.py (Corrected Layout)

import os
import torch
import chess
import pygame
import argparse
import threading
import warnings
import time

warnings.filterwarnings("ignore", message=".*pygame.*avx2.*")

from alphazero.env import ChessEnv
from alphazero.model import AlphaZeroNet
from alphazero.mcts import MCTS
from alphazero.move_encoder import MoveEncoder

# --- MODIFIED: New Layout Constants for a Robust UI ---
SQUARE_SIZE = 90
BOARD_SIZE = 8 * SQUARE_SIZE
BORDER_SIZE = 40
PANEL_WIDTH = 260
SCREEN_WIDTH = BOARD_SIZE + PANEL_WIDTH + BORDER_SIZE
SCREEN_HEIGHT = BOARD_SIZE + BORDER_SIZE

FONT_SIZE_PIECES = int(SQUARE_SIZE * 0.85)
FONT_SIZE_COORDS = 20
FONT_SIZE_MOVES = 22
FONT_SIZE_EVAL = 26
FONT_SIZE_STATUS = 48

# Colors
COLOR_LIGHT_SQ = (238, 238, 210)
COLOR_DARK_SQ = (118, 150, 86)
COLOR_BORDER = (40, 40, 40)
COLOR_PANEL = (50, 50, 50)
COLOR_TEXT = (220, 220, 220)
COLOR_HIGHLIGHT = (255, 255, 51, 150)
COLOR_VALID_MOVE = (82, 178, 204, 150)

class ChessGUI:
    def __init__(self, net, encoder, time_limit, device):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("AlphaZero Chess")

        # Create a separate surface for the board itself
        self.board_surface = pygame.Surface((BOARD_SIZE, BOARD_SIZE))

        # --- Fonts ---
        try:
            self.piece_font = pygame.font.SysFont("segoeuisymbol", FONT_SIZE_PIECES)
        except pygame.error:
            self.piece_font = pygame.font.Font(None, FONT_SIZE_PIECES)
        self.coord_font = pygame.font.SysFont("monospace", FONT_SIZE_COORDS, bold=True)
        self.move_font = pygame.font.SysFont("monospace", FONT_SIZE_MOVES)
        self.eval_font = pygame.font.SysFont("monospace", FONT_SIZE_EVAL, bold=True)
        self.status_font = pygame.font.SysFont("sans-serif", FONT_SIZE_STATUS, bold=True)

        self.net = net
        self.encoder = encoder
        self.mcts = MCTS(net, encoder, time_limit=time_limit, c_puct=1.41, device=device, dirichlet_alpha=0)
        self.human_color = chess.WHITE
        
        self.new_game_button = pygame.Rect(BOARD_SIZE + BORDER_SIZE + 30, SCREEN_HEIGHT - 80, 200, 50)
        self.unicode_map = {
            (chess.PAWN, chess.WHITE): "♙", (chess.PAWN, chess.BLACK): "♟",
            (chess.KNIGHT, chess.WHITE): "♘", (chess.KNIGHT, chess.BLACK): "♞",
            (chess.BISHOP, chess.WHITE): "♗", (chess.BISHOP, chess.BLACK): "♝",
            (chess.ROOK, chess.WHITE): "♖", (chess.ROOK, chess.BLACK): "♜",
            (chess.QUEEN, chess.WHITE): "♕", (chess.QUEEN, chess.BLACK): "♛",
            (chess.KING, chess.WHITE): "♔", (chess.KING, chess.BLACK): "♚",
        }
        self.reset_game()

    def reset_game(self):
        print("--- NEW GAME ---")
        self.env = ChessEnv(history_size=8)
        self.env.reset()
        self.selected_square = None
        self.valid_moves = []
        self.game_over = False
        self.ai_thread = None
        self.ai_is_thinking = False
        self.ai_move_result = None
        self.last_q_value = 0.0

    def _square_to_coords(self, square):
        """Converts a chess square (0-63) to pixel coords on the board surface."""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        if self.human_color == chess.BLACK:
            x = (7 - file) * SQUARE_SIZE
            y = rank * SQUARE_SIZE
        else:
            x = file * SQUARE_SIZE
            y = (7 - rank) * SQUARE_SIZE
        return x, y

    def _coords_to_square(self, x, y):
        """Converts pixel coords on the screen to a chess square."""
        # Adjust for the border offset
        x, y = x - BORDER_SIZE, y
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            return None
        
        file = x // SQUARE_SIZE
        rank = 7 - (y // SQUARE_SIZE)
        if self.human_color == chess.BLACK:
            return chess.square(7 - file, rank)
        else:
            return chess.square(file, rank)

    def _draw_all(self):
        """Main drawing function to render the entire game state."""
        self.screen.fill(COLOR_BORDER)
        self._draw_board_and_pieces()
        self._highlight_moves()
        # Blit the board surface onto the main screen with an offset for the border
        self.screen.blit(self.board_surface, (BORDER_SIZE, 0))
        self._draw_coordinates()
        self._draw_panel()
        if self.game_over:
            self._draw_game_over_message()
        pygame.display.flip()

    def _draw_board_and_pieces(self):
        """Draws the board and pieces onto the dedicated board_surface."""
        for r in range(8):
            for f in range(8):
                is_light = (r + f) % 2 == 1
                color = COLOR_LIGHT_SQ if is_light else COLOR_DARK_SQ
                rect = pygame.Rect(f * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(self.board_surface, color, rect)
        
        for square, piece in self.env.board.piece_map().items():
            char = self.unicode_map.get((piece.piece_type, piece.color))
            if char:
                color = (255, 255, 255) if piece.color == chess.WHITE else (0, 0, 0)
                text_surface = self.piece_font.render(char, True, color)
                x, y = self._square_to_coords(square)
                text_rect = text_surface.get_rect(center=(x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2))
                self.board_surface.blit(text_surface, text_rect)

    def _draw_coordinates(self):
        """Draws the rank and file labels in the border area."""
        for i in range(8):
            # Draw ranks (1-8)
            rank = str(8 - i) if self.human_color == chess.WHITE else str(i + 1)
            label = self.coord_font.render(rank, True, COLOR_TEXT)
            self.screen.blit(label, (BORDER_SIZE - 25, i * SQUARE_SIZE + 5))
            
            # Draw files (a-h)
            file = chess.FILE_NAMES[i] if self.human_color == chess.WHITE else chess.FILE_NAMES[7 - i]
            label = self.coord_font.render(file, True, COLOR_TEXT)
            self.screen.blit(label, (BORDER_SIZE + i * SQUARE_SIZE + (SQUARE_SIZE // 2 - 5), BOARD_SIZE + 5))

    def _draw_panel(self):
        panel_x = BOARD_SIZE + BORDER_SIZE
        panel_rect = pygame.Rect(panel_x, 0, PANEL_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_PANEL, panel_rect)

        # --- Evaluation Bar Logic ---
        # Map the Q-value from White's perspective (-1 to +1) to a 0-1 range
        white_bar_fraction = (self.last_q_value + 1) / 2.0
        
        eval_bar_total_height = 150
        # FIX: Ensure height is an integer for pygame.Rect
        white_height = int(eval_bar_total_height * white_bar_fraction)
        black_height = eval_bar_total_height - white_height
        
        white_rect = pygame.Rect(panel_x + 20, 50, 30, white_height)
        black_rect = pygame.Rect(panel_x + 20, 50 + white_height, 30, black_height)
        pygame.draw.rect(self.screen, (230, 230, 230), white_rect)
        pygame.draw.rect(self.screen, (20, 20, 20), black_rect)
        
        eval_text = f"{self.last_q_value:+.2f}"
        text_surf = self.eval_font.render(eval_text, True, COLOR_TEXT)
        self.screen.blit(text_surf, (panel_x + 60, 50))
        
        # --- Move History and Button Logic (unchanged) ---
        history_title = self.move_font.render("Move History:", True, COLOR_TEXT)
        self.screen.blit(history_title, (panel_x + 30, 220))
        
        move_y_start = 250
        available_height = self.new_game_button.top - move_y_start - 10
        max_moves_to_show = available_height // 25

        for i, move in enumerate(reversed(self.env.board.move_stack[-max_moves_to_show:])):
            move_num = (self.env.board.fullmove_number - i // 2)
            player = "w" if (len(self.env.board.move_stack) - i) % 2 != 0 else "b"
            move_text = f"{move_num}.{player}: {move.uci()}"
            text_surf = self.move_font.render(move_text, True, COLOR_TEXT)
            self.screen.blit(text_surf, (panel_x + 30, move_y_start + i * 25))

        pygame.draw.rect(self.screen, (80, 80, 90), self.new_game_button)
        button_text = self.eval_font.render("New Game", True, COLOR_TEXT)
        text_rect = button_text.get_rect(center=self.new_game_button.center)
        self.screen.blit(button_text, text_rect)

    def _highlight_moves(self):
        """Highlights moves on the board_surface."""
        if self.selected_square is not None:
            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            s.fill(COLOR_HIGHLIGHT)
            self.board_surface.blit(s, self._square_to_coords(self.selected_square))
            s.fill(COLOR_VALID_MOVE)
            for move in self.valid_moves:
                self.board_surface.blit(s, self._square_to_coords(move.to_square))
    
    def _draw_game_over_message(self):
        outcome = self.env.board.outcome()
        if not outcome: return
        winner = "Draw"
        if outcome.winner is not None:
            winner = "White Wins!" if outcome.winner == chess.WHITE else "Black Wins!"
        
        text_surf = self.status_font.render(winner, True, (255, 50, 50))
        text_rect = text_surf.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
        bg_rect = text_rect.inflate(40, 40)
        bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        bg_surf.fill((0, 0, 0, 180))
        self.screen.blit(bg_surf, bg_rect.move( (PANEL_WIDTH/2 - BORDER_SIZE, 0) ) )
        self.screen.blit(text_surf, text_rect.move( (PANEL_WIDTH/2 - BORDER_SIZE, 0) ) )

    def _handle_click(self, pos):
        # First, check if the "New Game" button was clicked
        if self.new_game_button.collidepoint(pos):
            self.reset_game()
            return

        # Use the corrected function call with the '*' unpack operator
        clicked_square = self._coords_to_square(*pos)
        if clicked_square is None: # Click was outside the board
            return

        # If a piece was already selected, check if this click is a valid move
        if self.selected_square is not None:
            move_to_make = next((m for m in self.valid_moves if m.to_square == clicked_square), None)
            if move_to_make:
                # Handle pawn promotions (auto-promote to queen for simplicity)
                if move_to_make.promotion and chess.square_rank(move_to_make.to_square) in [0, 7]:
                    move_to_make.promotion = chess.QUEEN
                self.env.step(move_to_make)
                self.selected_square, self.valid_moves = None, []
                return True # A move was successfully made

        # If no move was made, deselect any piece and try to select a new one
        self.selected_square, self.valid_moves = None, []
        piece = self.env.board.piece_at(clicked_square)
        if piece and piece.color == self.env.board.turn: # Can only select pieces of the current player
            self.selected_square = clicked_square
            self.valid_moves = [m for m in self.env.board.legal_moves if m.from_square == clicked_square]
        
        return False # No move was made

    def _run_ai_search(self):
        """The function that will be run in a separate thread."""
        root, _ = self.mcts.run(self.env)
        counts = torch.zeros(self.encoder.mapping_size, dtype=torch.float32)
        for m, child in root.children.items():
            counts[self.encoder.encode(m)] = child.N
        best_move = self.encoder.decode(int(counts.argmax().item()))
        self.ai_move_result = (best_move, root.Q)

    def run(self, human_color):
        """The main game loop."""
        self.human_color = human_color
        clock = pygame.time.Clock()
        running = True
        
        while running:
            is_human_turn = (self.env.board.turn == self.human_color) and not self.game_over

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN and is_human_turn:
                    self._handle_click(pygame.mouse.get_pos())

            if not is_human_turn and not self.game_over and not self.ai_is_thinking:
                print("Computer is thinking...")
                self.ai_is_thinking = True
                self.ai_thread = threading.Thread(target=self._run_ai_search, daemon=True)
                self.ai_thread.start()

            if self.ai_is_thinking and self.ai_thread and not self.ai_thread.is_alive():
                move, q_value = self.ai_move_result
                print(f"Computer plays: {move.uci()} (Q: {q_value:.3f})")

                # FIX: Correctly store the Q-value from White's perspective
                turn_before_move = self.env.board.turn
                self.last_q_value = q_value if turn_before_move == chess.WHITE else -q_value
                
                self.env.step(move)
                
                self.ai_is_thinking = False
                self.ai_thread = None
                self.ai_move_result = None

            if not self.game_over and self.env.board.is_game_over():
                self.game_over = True
                print("GAME OVER")

            self._draw_all()
            clock.tick(30)

        pygame.quit()

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model onto device: {device}")
    net = AlphaZeroNet(in_channels=119).to(device)
    if os.path.exists(args.checkpoint):
        net.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        print(f"Loaded weights from '{args.checkpoint}'")
    else:
        print(f"ERROR: Checkpoint not found at '{args.checkpoint}'.")
        return
    net.eval()
    encoder = MoveEncoder()
    human_color = chess.WHITE if args.color.lower() == 'white' else chess.BLACK
    gui = ChessGUI(net, encoder, args.time_limit, device)
    gui.run(human_color)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a graphical game of chess against the AlphaZero model.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth", help="Path to the model checkpoint.")
    parser.add_argument("--time-limit", type=int, default=2, help="Time in seconds for the AI to think per move.")
    parser.add_argument("--color", type=str, default="white", choices=['white', 'black'], help="The color you want to play as.")
    args = parser.parse_args()
    main(args)