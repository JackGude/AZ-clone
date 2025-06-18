# alphazero/utils.py
import os
import random
import shutil
import pandas as pd
import csv
from .env import ChessEnv
from config import (
    OPENINGS_SELFPLAY_PATH,
    OPENINGS_EVAL_PATH,
    REPLAY_DIR,
    MAX_GAMES_IN_BUFFER,
    AUTOMATE_WIN_THRESHOLD,
    CANDIDATE_MODEL_PATH,
    BEST_MODEL_PATH,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Load Self-Play Openings
# ─────────────────────────────────────────────────────────────────────────────
try:
    SELFPLAY_OPENINGS = pd.read_csv(OPENINGS_SELFPLAY_PATH)
    # print(f"Successfully loaded {len(SELFPLAY_OPENINGS)} openings for self-play.")
except FileNotFoundError:
    print(
        f"Warning: {OPENINGS_SELFPLAY_PATH} not found. Self-play will start from the standard position."
    )
    SELFPLAY_OPENINGS = None

# ─────────────────────────────────────────────────────────────────────────────
#  Load Evaluation Openings
# ─────────────────────────────────────────────────────────────────────────────
EVALUATION_OPENINGS = []
try:
    with open(OPENINGS_EVAL_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            EVALUATION_OPENINGS.append((row["name"], row["fen"]))
    # print(f"Successfully loaded {len(EVALUATION_OPENINGS)} openings for evaluation.")
except FileNotFoundError:
    print(
        f"Warning: {OPENINGS_EVAL_PATH} not found. Evaluation will require manual FENs or start from the standard position."
    )

# ─────────────────────────────────────────────────────────────────────────────
#  Utility Functions
# ─────────────────────────────────────────────────────────────────────────────


def format_time(seconds):
    """Formats a time in seconds into a human-readable string."""
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def setup_selfplay_opening(env: ChessEnv, game_num: int = 0):
    """
    Sets up the opening position for a self-play game using the env's own method.
    Has a 75% chance to use the opening book, otherwise uses standard start.
    """
    prefix = f"[Worker {game_num + 1}]"
    opening_name = "Standard Opening"

    if (
        SELFPLAY_OPENINGS is not None
        and not SELFPLAY_OPENINGS.empty
        and random.random() < 0.75
    ):
        random_opening = SELFPLAY_OPENINGS.sample(n=1).iloc[0]
        move_string = random_opening["moves"]
        opening_name = random_opening.get("name", "Unknown Opening")

        env.set_board_from_moves(move_string)

    print(f"{prefix} Starting from opening: {opening_name}", flush=True)


def manage_replay_buffer():
    """
    Keeps a maximum number of recent DRAW games, while preserving all WIN/LOSS games.
    """
    try:
        all_game_files = [
            os.path.join(REPLAY_DIR, f)
            for f in os.listdir(REPLAY_DIR)
            if f.endswith(".pkl")
        ]
        draw_files = [
            f for f in all_game_files if "draw" in os.path.basename(f).lower()
        ]
        draw_files.sort(key=os.path.getmtime)

        while len(draw_files) > MAX_GAMES_IN_BUFFER:
            file_to_delete = draw_files.pop(0)
            os.remove(file_to_delete)
    except Exception as e:
        print(f"Warning: Could not manage replay buffer. Error: {e}", flush=True)


def promote_candidate(win_rate):
    """
    If the candidate's win rate is above the threshold, it replaces 'best.pth'.
    """
    if win_rate >= AUTOMATE_WIN_THRESHOLD:
        print(
            f"→ Candidate met win threshold ({win_rate:.2f} >= {AUTOMATE_WIN_THRESHOLD}). Promoting to 'best.pth'."
        )
        # Use rename for an atomic operation, replacing the old best model.
        shutil.move(CANDIDATE_MODEL_PATH, BEST_MODEL_PATH)
        return True
    else:
        print(
            f"→ Candidate failed to meet win threshold ({win_rate:.2f} < {AUTOMATE_WIN_THRESHOLD}). Discarding."
        )
        os.remove(CANDIDATE_MODEL_PATH)
        return False