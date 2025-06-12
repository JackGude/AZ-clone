# AlphaZero Chess Engine

A complete, end-to-end implementation of an AlphaZero-style chess engine in Python. This project includes self-play data generation, a training pipeline with PyTorch, model evaluation, hyperparameter tuning with Weights & Biases, and a playable GUI.

## Features

* **Model:** SE-ResNet (Residual Network with Squeeze-and-Excitation blocks) implemented in PyTorch.
* **Search:** Monte Carlo Tree Search (MCTS) with time management.
* **Training:** Fully automated training loop manages self-play, training, and evaluation cycles.
* **Experiment Tracking:** Integrated with Weights & Biases for real-time metric visualization, hyperparameter sweeps, and model artifact versioning.
* **Data Pipeline:** Scalable replay buffer that stores games as individual files to handle massive datasets without RAM issues.
* **GUI:** A full-featured graphical interface built with Pygame to play against the trained model.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
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

## How to Use

### Training a Model
The entire pipeline can be run with the automation script:
```bash
python automate.py
```
To resume from a specific generation:
```bash
python automate.py --start-generation 15
```

### Playing Against the Model
To play a game against the current best model using the GUI:
```bash
# Play as White (default)
python gui_play.py

# Play as Black and give the AI 10 seconds per move
python gui_play.py --color black --time-limit 10
```

### Downloading Models from Weights & Biases
To download a trained model from Weights & Biases:

1. **List available models:**
   ```bash
   wandb artifact ls alphazero-chess
   ```

2. **Download a specific model:**
   ```bash
   wandb artifact get alphazero-chess/model-<ID>:v0 --root ./checkpoints/download/
   ```
   Replace `<ID>` with the model ID from the list. The model will be downloaded to the `checkpoints/download` directory.

3. **Use the downloaded model:**
   The model will be available in the `checkpoints/download` directory and can be used with the GUI or evaluation scripts.

Note: You need to be logged into Weights & Biases (`wandb login`) and have access to the project to download models.