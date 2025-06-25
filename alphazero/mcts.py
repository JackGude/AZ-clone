# mcts.py

import math
import torch
import numpy as np
import time


class TreeNode:
    """A node in the Monte Carlo Search Tree. This version is simplified and does not contain virtual loss."""

    def __init__(self, parent=None, prior=1.0):
        self.parent = parent
        self.children = {}
        self.P = prior  # Prior probability
        self.N = 0  # Visit count
        self.W = 0.0  # Total action value
        self.Q = 0.0  # Mean action value

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, priors):
        for move, p in priors.items():
            if move not in self.children:
                self.children[move] = TreeNode(parent=self, prior=p)

    def select(self, c_puct):
        """Selects the child with the highest PUCT score."""
        best_score, best_move, best_child = -float("inf"), None, None
        for move, child in self.children.items():
            # The PUCT formula balances exploitation (Q) and exploration (U)
            U = c_puct * child.P * math.sqrt(self.N) / (1 + child.N)
            score = child.Q + U
            if score > best_score:
                best_score, best_move, best_child = score, move, child
        return best_move, best_child

    def backup(self, value):
        """Backpropagates the evaluation result up the tree."""
        node = self
        while node is not None:
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            value = -value  # Negate value for the parent (opponent's turn)
            node = node.parent


class MCTS:
    """A single-threaded, batched MCTS implementation. This version is significantly
    faster as it avoids using deepcopy in the main simulation loop."""

    def __init__(
        self,
        net,
        encoder,
        time_limit=5,
        c_puct=1.41,
        use_dynamic_cpuct: bool = False,
        device="cpu",
        batch_size=64,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    ):
        self.net = net.to(device).eval()
        self.encoder = encoder
        self.time_limit = time_limit
        self.c_puct = c_puct
        self.use_dynamic_cpuct = use_dynamic_cpuct
        self.device = device
        self.batch_size = batch_size
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    @torch.no_grad()  # Disables gradient calculation for all operations for performance.
    def run(self, env):
        root = TreeNode()

        # 1. Initial Evaluation & Expansion of the Root Node
        policy_logits, value = self.net(env.get_state_tensor().to(self.device))
        root_value = value.item()  # Cache the raw network eval of the root

        legal_moves = {move for move in env.board.legal_moves}
        probs = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        priors = {move: probs[self.encoder.encode(move)] for move in legal_moves}

        # Add Dirichlet noise to the root's priors for exploration
        if self.dirichlet_alpha > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(priors))
            for i, move in enumerate(priors.keys()):
                priors[move] = (1 - self.dirichlet_epsilon) * priors[
                    move
                ] + self.dirichlet_epsilon * noise[i]

        root.expand(priors)

        final_c_puct = self.c_puct # Default to the fixed value
        if self.use_dynamic_cpuct:
            if abs(root_value) < 0.1:
                final_c_puct = 2.0  # More exploration in drawn positions
            elif abs(root_value) > 0.5:
                final_c_puct = 1.0  # More exploitation in decisive positions  

        # Main simulation loop - continues until the time limit is reached.
        start_time = time.time()
        while time.time() - start_time < self.time_limit:

            leaf_nodes_to_evaluate = []

            # Run a batch of selections to find leaf nodes.
            for _ in range(self.batch_size):
                # Check the time on every simulation, not just per batch.
                if time.time() - start_time >= self.time_limit:
                    break # Exit the inner for-loop immediately

                node = root
                # Use board.copy(), which is much faster than deepcopying the whole environment.
                board_copy = env.board.copy()

                # a) Selection: Traverse the tree from the root to a leaf node.
                while not node.is_leaf():
                    move, node = node.select(final_c_puct)
                    board_copy.push(move)

                # b) If the game ends during selection, backup the true result immediately.
                if board_copy.is_game_over():
                    outcome = board_copy.outcome()
                    # The value is from the perspective of the player *whose turn it is now*.
                    value = (
                        0.0
                        if outcome.winner is None
                        else (-1.0 if outcome.winner == board_copy.turn else 1.0)
                    )
                    node.backup(value)
                else:
                    # c) Otherwise, add the non-terminal leaf node to our batch to be evaluated.
                    leaf_nodes_to_evaluate.append((node, board_copy))

            if not leaf_nodes_to_evaluate:
                continue

            # 2. Batch Evaluation: Process all collected leaf nodes in a single NN forward pass for efficiency.
            states = [
                env.get_state_tensor(board) for _, board in leaf_nodes_to_evaluate
            ]
            state_batch = torch.cat(states).to(self.device)
            policy_logits_b, values_b = self.net(state_batch)

            probs_b = torch.softmax(policy_logits_b, dim=1).cpu().numpy()
            values_np = values_b.squeeze(1).cpu().numpy()

            # 3. Batch Expansion and Backup: Distribute the NN results back to the leaf nodes.
            for i, (node, board) in enumerate(leaf_nodes_to_evaluate):
                legal_moves = {move for move in board.legal_moves}
                priors = {
                    move: probs_b[i][self.encoder.encode(move)] for move in legal_moves
                }
                node.expand(priors)
                node.backup(values_np[i])

        return root, root_value
