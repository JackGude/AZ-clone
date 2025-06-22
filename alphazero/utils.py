# alphazero/utils.py

# -----------------------------------------------------------------------------
#  Standard Library Imports
# -----------------------------------------------------------------------------
import os
import sys


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
