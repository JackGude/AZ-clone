# AlphaZero Chess Engine

[](https://github.com/) [](https://opensource.org/licenses/MIT)

A complete, end-to-end implementation of an AlphaZero-style chess engine in Python. This project includes self-play data generation, a robust training pipeline, deterministic model evaluation, and a full-featured GUI to play against the trained models.

## Key Features

  * **Advanced Model Architecture:** A deep residual network (ResNet) with Squeeze-and-Excitation (SE) blocks, implemented in PyTorch, provides a strong foundation for learning.
  * **Efficient Search Algorithm:** A batched Monte Carlo Tree Search (MCTS) implementation maximizes GPU throughput. The search incorporates a PUCT formula and Dirichlet noise to balance exploitation and exploration effectively.
  * **Automated End-to-End Training:** The entire pipeline is orchestrated by a central script that manages the continuous cycle of self-play, training, and evaluation.
  * **Comprehensive Experiment Tracking:** Deeply integrated with Weights & Biases for real-time metric visualization, hyperparameter sweeps, and model artifact versioning.
  * **Robust Data Pipeline:** Generates and stores game data in a scalable replay buffer. Games are saved as individual files, allowing the system to handle massive datasets without being limited by system RAM.
  * **Deterministic Evaluation:** Head-to-head matches between models are run with deterministic settings (no randomness) and a curated opening book to ensure fair and reproducible comparisons.
  * **Playable GUI:** A full-featured graphical interface built with Pygame allows you to play directly against any trained model.

## Project Architecture

The engine's learning process is driven by three core pipeline components, orchestrated by the main `automate.py` script.

1.  **`self_play.py` (Data Generation):** The current best model plays games against itself to generate high-quality training data.
2.  **`train.py` (Model Training):** A new "candidate" model is trained on the data collected from the replay buffer. This script can be used for standard training or as part of a Weights & Biases hyperparameter sweep.
3.  **`evaluate.py` (Model Evaluation):** The new candidate model plays a head-to-head match against the current best model. If the candidate wins by a sufficient margin, it is promoted to become the new "best."

This cycle repeats indefinitely, allowing the model to gradually improve over many generations.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone JackGude/AZ-clone
    cd AZ-clone
    ```
2.  **Create and activate a conda environment:**
    ```bash
    conda create --name alphazero python=3.10
    conda activate alphazero
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

This project is designed to be run either through the end-to-end automation script or by executing the individual components manually for debugging, analysis, or specific tasks.

### Automated Training (Primary Method)

To run the entire training loop, use the main automation script. This will continuously generate data, train new models, and evaluate them.

```bash
python automate.py
```

To resume the training loop from a specific generation number:

```bash
python automate.py --start-generation 15
```

### Manual Execution of Pipeline Components

Each core script can be run by hand. This is useful for debugging, testing a single component, or running specialized jobs like a hyperparameter sweep.

**Common Argument:** All scripts accept a `--no-wandb` flag to disable Weights & Biases logging for a local or test run.

#### 1\. Generating Self-Play Data

To manually generate a specific number of games using the current best model:

```bash
python self_play.py --num_games 100 --gen-id "manual-selfplay" --result-file "selfplay_results.json"
```

#### 2\. Training a Model

The training script can be used to train a single model on the existing data in the `replay_buffer`. It also serves as the target for W\&B sweeps.

  * **To run a single training session:**
    ```bash
    python train.py --gen-id "manual-training" --result-file "training_results.json"
    ```
  * **To run a hyperparameter sweep (using W\&B):**
    1.  Define your sweep configuration in a `sweep.yaml` file.
    2.  Initialize the sweep with `wandb sweep sweep.yaml`.
    3.  Run the wandb agent, which will repeatedly call `train.py`:
        ```bash
        wandb agent <your-sweep-id>
        ```

#### 3\. Evaluating Two Models

To run a head-to-head match between two specific model checkpoints:

```bash
python evaluate.py --old "checkpoints/best.pth" --new "checkpoints/candidate.pth" --games 50 --gen-id "manual-eval" --result-file "eval_results.json"
```

### Playing Against the Model

To play a game against the current best model using the GUI:

```bash
# Play as White (default)
python gui_play.py

# Play as Black and give the AI 10 seconds per move
python gui_play.py --color black --time-limit 10
```

### Working with Models from Weights & Biases

You can easily download models that have been saved as artifacts in Weights & Biases.

1.  **List available models in the project:**

    ```bash
    wandb artifact ls alphazero-chess
    ```

2.  **Download a specific model version:**

    ```bash
    wandb artifact get alphazero-chess/model-<RUN_ID>:v0 --root ./checkpoints/download/
    ```

    Replace `<RUN_ID>` with the ID of the run from the list. The model (`best.pth`) will be downloaded to the specified directory.

3.  **Use the downloaded model:**
    The downloaded model can be used with the GUI or evaluation scripts by pointing to its file path.

*Note: You must be logged into your W\&B account (`wandb login`) and have access to the project to download artifacts.*

## Configuration

All major hyperparameters, file paths, and training parameters are centralized in `config.py`. This file is the single source of truth for configuring the behavior of all components in the pipeline.