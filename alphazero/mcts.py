# mcts.py

import math
import torch
import numpy as np
import time
from torch.amp import autocast
from copy import deepcopy
from .state_encoder import encode_history

# ─────────────────────────────────────────────────────────────────────────────
#  Tree Node
# ─────────────────────────────────────────────────────────────────────────────

class TreeNode:
    """A node in the Monte Carlo Search Tree."""

    def __init__(self, parent=None, prior=1.0):
        """
        Initializes a TreeNode.

        Args:
            parent (TreeNode, optional): The parent of this node. Defaults to None.
            prior (float, optional): The prior probability of selecting this node from its parent.
                                     This is determined by the policy head of the neural network.
        """
        self.parent = parent
        self.children = {}  # A map from move to TreeNode

        # --- MCTS Statistics ---
        self.P = prior          # Prior probability of this node
        self.N = 0              # Visit count
        self.W = 0.0            # Total action value (accumulated from child node values)
        self.Q = 0.0            # Mean action value (W / N)

        # --- Enhancements for Parallel MCTS ---
        # Virtual loss is added to a node when it's selected in a simulation.
        # This discourages other parallel simulations from picking the exact same path
        # until the first simulation's result is backed up.
        self.virtual_loss = 0

    def is_leaf(self):
        """Checks if the node is a leaf node (i.e., has no children)."""
        return len(self.children) == 0

    def expand(self, priors):
        """
        Expands a leaf node by creating children for all legal moves.

        Args:
            priors (dict): A dictionary mapping each legal move to its prior probability
                           from the neural network's policy head.
        """
        for move, p in priors.items():
            if move not in self.children:
                self.children[move] = TreeNode(parent=self, prior=p)

    def select(self, c_puct):
        """
        Selects the child node with the highest Upper Confidence Bound for Trees (UCT) score.
        This balances exploration (visiting promising but less-visited nodes) and
        exploitation (visiting nodes known to have high values).

        Args:
            c_puct (float): A constant controlling the level of exploration.

        Returns:
            tuple: A tuple containing (best_move, best_child_node).
        """
        best_move, best_child, best_score = None, None, -float('inf')
        total_parent_visits = self.N + self.virtual_loss

        for move, child in self.children.items():
            # Adjust child's statistics with virtual loss for the selection formula.
            # This makes the node appear less promising to other concurrent searches.
            child_visits = child.N + child.virtual_loss
            child_q_value = (child.W - child.virtual_loss) / (child_visits if child_visits > 0 else 1)

            # The PUCT (Polynomial Upper Confidence Trees) formula from AlphaZero
            # Exploration term: Encourages visiting nodes with high prior P and low visit count N.
            U = c_puct * child.P * math.sqrt(total_parent_visits) / (1 + child_visits)
            
            # The score is a combination of the node's mean value (exploitation) and the exploration term.
            score = child_q_value + U

            if score > best_score:
                best_move, best_child, best_score = move, child, score
        
        return best_move, best_child

    def backup(self, value):
        """
        Backpropagates the evaluation result up the tree from this node to the root.

        Args:
            value (float): The value of the current state, as determined by the value head of the
                           network or the outcome of the game. This value is always from the
                           perspective of the player whose turn it is at the current node.
        """
        node = self
        while node is not None:
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            
            # The value must be negated for the parent node, as it represents the opponent's perspective.
            # A good move for one player is a bad move for the other.
            value = -value
            node = node.parent
            
    def add_virtual_loss(self, amount=1):
        """Propagates virtual loss up the tree during the selection phase."""
        self.virtual_loss += amount
        if self.parent:
            self.parent.add_virtual_loss(amount)

    def remove_virtual_loss(self, amount=1):
        """Removes virtual loss after the true value has been backed up."""
        self.virtual_loss -= amount
        if self.parent:
            self.parent.remove_virtual_loss(amount)

# ─────────────────────────────────────────────────────────────────────────────
#  MCTS Class
# ─────────────────────────────────────────────────────────────────────────────

class MCTS:
    """Monte Carlo Tree Search algorithm implementation."""

    def __init__(self, net, encoder, time_limit=5, c_puct=1.41, device="cpu", batch_size=64, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        self.net = net.to(device).eval()
        self.encoder = encoder
        self.time_limit = time_limit # Store time limit in seconds
        self.c_puct = c_puct
        self.device = device
        self.is_cuda = (device == "cuda")
        self.batch_size = batch_size
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def run(self, env):
        """
        Runs the MCTS search from a given environment state.

        Args:
            env (ChessEnv): The current game environment.

        Returns:
            tuple: (root_node, root_value) where root_node contains the search statistics
                   and root_value is the raw network evaluation of the root state.
        """
        # 1. Create the root node of the search tree
        root = TreeNode(parent=None, prior=1.0)
        state_tensor = self._env_to_tensor(env)

        # 2. Evaluate the root node with the neural network
        if self.device.startswith("cuda"):
            state_tensor = state_tensor.to(dtype=torch.float16)
            with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
                logits, value = self.net(state_tensor)
        else:
            with torch.no_grad():
                logits, value = self.net(state_tensor)
        
        # Cache the raw network value of the root for resignation checks
        root_value = value.item()
        
        # Get prior probabilities for all legal moves from the policy head
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        legal_moves = env.legal_moves()
        priors = {m: probs[self.encoder.encode(m)] for m in legal_moves}
        
        # 3. Add Dirichlet noise to the root's priors to encourage exploration
        if self.dirichlet_alpha > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(priors))
            for i, move in enumerate(priors.keys()):
                priors[move] = (1 - self.dirichlet_epsilon) * priors[move] + self.dirichlet_epsilon * noise[i]
        
        # 4. Expand the root node
        root.expand(priors)

        # 5. Main simulation loop
        pending_nodes, pending_envs = [], []
        start_time = time.time()
        
        while time.time() - start_time < self.time_limit:
            # Run a batch of simulations
            for _ in range(self.batch_size):
                # Check time limit inside the inner loop for more responsiveness
                if time.time() - start_time > self.time_limit:
                    break
                
                sim_env = deepcopy(env)
                node = root
                node.add_virtual_loss()

                while not node.is_leaf():
                    move, node = node.select(self.c_puct)
                    _, _, done = sim_env.step(move)
                    node.add_virtual_loss()
                    if done:
                        outcome = sim_env.board.outcome()
                        value = 0.0 if outcome.winner is None else -1.0
                        node.backup(value)
                        node.remove_virtual_loss()
                        break # Break from inner 'while not node.is_leaf()'
                else: # This 'else' belongs to the 'while not node.is_leaf()'
                    pending_nodes.append(node)
                    pending_envs.append(sim_env)
            
            # Flush the batch for network evaluation
            if pending_nodes:
                self._batch_expand_and_backup(pending_nodes, pending_envs)
                pending_nodes.clear()
                pending_envs.clear()
        
        # Final check to ensure at least one simulation was run for the root
        if root.N == 0:
            # If time was too short, run one quick batch to get some stats
            self._batch_expand_and_backup([root], [deepcopy(env)])

        return root, root_value

    def _batch_expand_and_backup(self, nodes, envs):
        """
        Evaluates a batch of leaf nodes with the network, then expands and backs them up.
        """
        # Create a batch of state tensors from the environments
        state_tensors = [self._env_to_tensor(e) for e in envs]
        x_batch = torch.cat(state_tensors, dim=0).to(self.device)

        # Evaluate the batch
        if self.device.startswith("cuda"):
            x_batch = x_batch.to(dtype=torch.float16)
            with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
                logits_b, values_b = self.net(x_batch)
        else:
            with torch.no_grad():
                logits_b, values_b = self.net(x_batch)

        probs_b = torch.softmax(logits_b, dim=1).cpu().numpy()
        values_np = values_b.squeeze(1).cpu().numpy()

        # For each node in the processed batch...
        for i, node in enumerate(nodes):
            value = values_np[i]
            legal_moves = envs[i].legal_moves()
            priors = {m: probs_b[i][self.encoder.encode(m)] for m in legal_moves}
            
            # d) Expansion & Backup
            node.expand(priors)
            node.backup(value)
            
            # Evaluation and backup are done, so remove the virtual loss
            node.remove_virtual_loss()

    def _env_to_tensor(self, env):
        """Converts an environment's history into a state tensor for the network."""
        hist = list(env.history)
        x = encode_history(hist, history_size=env.history_size)
        # This ensures the tensor is moved to the same device as the model (e.g., 'cuda:0')
        return x.unsqueeze(0).to(self.device)