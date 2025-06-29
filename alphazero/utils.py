# alphazero/utils.py

# -----------------------------------------------------------------------------
#  Standard Library Imports
# -----------------------------------------------------------------------------
import os
import sys
import torch
from alphazero.model import AlphaZeroNet

# ─────────────────────────────────────────────────────────────────────────────
#  Utility Functions
# ─────────────────────────────────────────────────────────────────────────────


def ensure_project_root():
    """
    Checks if the script is being run from the project's root directory.
    If not, it prints an error and exits.
    """
    # --- Sanity Check: Ensure script is run from project root ---
    EXPECTED_DIRS = ["alphazero", "pipeline", "tools"]
    missing_dirs = [d for d in EXPECTED_DIRS if not os.path.isdir(d)]

    if missing_dirs:
        print(
            "Error: This script must be run from the project's root directory.",
            file=sys.stderr,
        )
        print(
            f"The current directory '{os.getcwd()}' is missing the following required subdirectories: {', '.join(missing_dirs)}",
            file=sys.stderr,
        )
        sys.exit(1)


def format_time(seconds):
    """Formats a time in seconds into a human-readable string."""
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def load_or_initialize_model(model_path, verbose=False, *args, **kwargs):
    """
    Load a model from the given path, or initialize a new one if it doesn't exist.
    
    Args:
        model_path (str): Path to the model file
        *args, **kwargs: Arguments to pass to the model class constructor
        
    Returns:
        tuple: (model, was_initialized) where was_initialized is True if a new model was created
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroNet(*args, **kwargs).to(device)
    was_initialized = False
    
    try:
        # Try to load the model
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            if verbose:
                print(f"Loaded model from {model_path}", flush=True)
        else:
            was_initialized = True
            if verbose:
                print(f"Model not found at {model_path}, initializing and saving new model", flush=True)
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
            # Save the newly initialized model
            torch.save(model.state_dict(), model_path)
    except Exception as e:
        was_initialized = True
        if verbose:
            print(f"Error loading model from {model_path}, initializing new model. Error: {str(e)}", flush=True)
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        # Save the newly initialized model
        torch.save(model.state_dict(), model_path)
    
    model.eval()
    return model, was_initialized
