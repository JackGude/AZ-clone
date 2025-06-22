# config.py

import os

# -----------------------------------------------------------------------------
#  Project and File Paths
# -----------------------------------------------------------------------------
PROJECT_NAME = "alphazero-chess"
NUM_WORKERS = 4

TENSOR_CACHE_DIR = "replay_cache"
MODEL_DIR = "models"
EVAL_GAMES_DIR = "eval_games"
LOGS_DIR = "logs"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best.pth")
CANDIDATE_MODEL_PATH = os.path.join(MODEL_DIR, "candidate.pth")

OPENINGS_SELFPLAY_PATH = "openings/selfplay_openings.csv"
OPENINGS_EVAL_PATH = "openings/evaluation_openings.csv"

SELFPLAY_SCRIPT = "pipeline/play.py"
TRAIN_SCRIPT = "pipeline/train.py"
EVALUATE_SCRIPT = "pipeline/eval.py"
STOP_FILE = "stop.txt"  # If this file exists, the automation will stop after the current generation

# -----------------------------------------------------------------------------
#  Automation Pipeline Config (automate.py)
# -----------------------------------------------------------------------------
AUTOMATE_NUM_SELFPLAY_GAMES = 600  # Number of self-play games to generate per generation
AUTOMATE_NUM_EVAL_GAMES = 48  # Number of games to play to compare models
AUTOMATE_EVAL_TIME_LIMIT = 10  # Time in seconds per move for evaluation games
AUTOMATE_WIN_THRESHOLD = 0.52  # Win rate needed for a candidate to be promoted
AUTOMATE_WARMUP_GENS = 15  # Number of initial generations to run without evaluation

# -----------------------------------------------------------------------------
#  Self-Play Config (self_play.py)
# -----------------------------------------------------------------------------
NUM_SELFPLAY_WORKERS = NUM_WORKERS  # Number of workers to use for self-play
DEFAULT_NUM_SELFPLAY_GAMES = 4
MAX_FILES_IN_BUFFER = 1_000_000  # Max replay buffer size

# MCTS Parameters
SELFPLAY_CPUCT = 1.41  # PUCT constant for MCTS exploration
DIRICHLET_EPSILON = 0.25  # Weight of Dirichlet noise
DIRICHLET_ALPHA = 0.25  # Shape of Dirichlet noise
TEMP_THRESHOLD = 30  # Number of moves to use temperature sampling

# Game Termination Parameters
SELFPLAY_MAX_MOVES = 200  # Cap for game length in self-play
RESIGN_THRESHOLD = 0.95  # Resign if win probability is below (1 - threshold)
DRAW_CAP_PENALTY = 0.0  # Reward for hitting the move cap

# -----------------------------------------------------------------------------
#  Training Config (train.py)
# -----------------------------------------------------------------------------
NUM_TRAINING_WORKERS = NUM_WORKERS  # Number of DataLoader workers to use for training
# Hyperparameters from your latest sweep
WARMUP_LEARNING_RATE = 1.009e-4
WARMUP_WEIGHT_DECAY = 3.027e-5
LEARNING_RATE = 6.181e-5
WEIGHT_DECAY = 4.731e-5

# Training Process Parameters
TRAIN_WINDOW_SIZE = 400_000  # Number of positions to sample for training
BATCH_SIZE = 1536
MAX_EPOCHS = 20
PATIENCE = 3  # Early stopping patience

# -----------------------------------------------------------------------------
#  Evaluation Config (evaluate.py)
# -----------------------------------------------------------------------------
NUM_EVAL_WORKERS = NUM_WORKERS  # Number of workers to use for evaluation
DEFAULT_NUM_EVAL_GAMES = 6
DEFAULT_EVAL_TIME_LIMIT = 2  # Time in seconds per move for evaluation games
DEFAULT_EVAL_CPUCT = 1.0  # PUCT constant for evaluation MCTS

# Adjudication Parameters
ADJUDICATION_START_MOVE = 40 # Don't check for draws before this move
DRAW_ADJUDICATION_THRESHOLD = 0.08  # Adjudicate if abs(eval) is below this value
DRAW_ADJUDICATION_PATIENCE = 12  # For this many consecutive moves
WIN_ADJUDICATION_THRESHOLD = 0.90  # Adjudicate if eval is above this value
WIN_ADJUDICATION_PATIENCE = 6  # For this many consecutive moves

# -----------------------------------------------------------------------------
#  Stockfish Analysis Config (stockfish_analysis.py)
# -----------------------------------------------------------------------------
STOCKFISH_PATH = "models/stockfish/stockfish-16-x64-avx2.exe"
NUM_STOCKFISH_WORKERS = NUM_WORKERS  # Number of workers to use for stockfish analysis
