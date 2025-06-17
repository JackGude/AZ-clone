# config.py

import os

# -----------------------------------------------------------------------------
#  Project and File Paths
# -----------------------------------------------------------------------------

PROJECT_NAME = "alphazero-chess"

REPLAY_DIR = "replay_buffer"
CHECKPOINT_DIR = "checkpoints"
LOGS_DIR = "logs"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best.pth")
CANDIDATE_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "candidate.pth")

OPENINGS_SELFPLAY_PATH = "openings/selfplay_openings.csv"
OPENINGS_EVAL_PATH = "openings/evaluation_openings.py"

SELFPLAY_SCRIPT    = "self_play.py"
TRAIN_SCRIPT       = "train.py"
EVALUATE_SCRIPT    = "evaluate.py"
STOP_FILE          = "stop.txt"     # If this file exists, the automation will stop after the current generation

# -----------------------------------------------------------------------------
#  Automation Pipeline Config (automate.py)
# -----------------------------------------------------------------------------
AUTOMATE_NUM_SELFPLAY_GAMES = 100    # Number of self-play games to generate per generation
AUTOMATE_NUM_EVAL_GAMES = 24         # Number of games to play to compare models
AUTOMATE_WIN_THRESHOLD = 0.52        # Win rate needed for a candidate to be promoted
AUTOMATE_WARMUP_GENS = 15            # Number of initial generations to run without evaluation
AUTOMATE_EVAL_TIME_LIMIT = 20        # Time in seconds per move for evaluation games

# -----------------------------------------------------------------------------
#  Self-Play Config (self_play.py)
# -----------------------------------------------------------------------------
DEFAULT_NUM_SELFPLAY_GAMES = 50
MAX_GAMES_IN_BUFFER = 500   # Max replay buffer size (for draws)

# MCTS Parameters
SELFPLAY_CPUCT = 1.41       # PUCT constant for MCTS exploration
DIRICHLET_EPSILON = 0.25    # Weight of Dirichlet noise
DIRICHLET_ALPHA = 0.25      # Shape of Dirichlet noise
TEMP_THRESHOLD = 30         # Number of moves to use temperature sampling

# Game Termination Parameters
SELFPLAY_MAX_MOVES = 200    # Cap for game length in self-play
RESIGN_THRESHOLD = 0.90     # Resign if win probability is below (1 - threshold)
DRAW_CAP_PENALTY = 0.0      # Reward for hitting the move cap

# -----------------------------------------------------------------------------
#  Training Config (train.py)
# -----------------------------------------------------------------------------
# Hyperparameters from your latest sweep
LEARNING_RATE = 1.009e-4
WEIGHT_DECAY = 3.027e-5

# Training Process Parameters
TRAIN_WINDOW_SIZE = 100_000 # Number of positions to sample for training
BATCH_SIZE = 1536
MAX_EPOCHS = 20
PATIENCE = 3                # Early stopping patience

# -----------------------------------------------------------------------------
#  Evaluation Config (evaluate.py)
# -----------------------------------------------------------------------------
DEFAULT_NUM_EVAL_GAMES = 24
DEFAULT_EVAL_TIME_LIMIT = 10        # Time in seconds per move for evaluation games
DEFAULT_EVAL_CPUCT = 1.0            # PUCT constant for evaluation MCTS

# Adjudication Parameters
DRAW_ADJUDICATION_THRESHOLD = 0.08  # Adjudicate if abs(eval) is below this value
DRAW_ADJUDICATION_PATIENCE = 12     # For this many consecutive moves
WIN_ADJUDICATION_THRESHOLD = 0.90   # Adjudicate if eval is above this value
WIN_ADJUDICATION_PATIENCE = 6       # For this many consecutive moves