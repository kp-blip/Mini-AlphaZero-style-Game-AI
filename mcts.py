"""
mcts.py
-------
AlphaZero-style Monte Carlo Tree Search: instead of random rollouts (classic
MCTS), leaf nodes are evaluated directly by the neural network, and the tree
policy uses the PUCT formula (Predictor + UCT) so the network's policy head
biases which moves get explored.

    PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

After `num_sims` simulations from the root, the move-visit-counts N(s, ·)
give an improved policy `pi`, which is both:
  1) used to pick the next move during self-play / play, and
  2) used as the training TARGET for the network's policy head.
"""

import math
import numpy as np


class MCTS:
    def __init__(self, game, net, c_puct=1.5, num_sims=100, dirichlet_alpha=0.3,
                 dirichlet_eps=0.25):
        self.game = game
        self.net = net
        self.c_puct = c_puct
        self.num_sims = num_sims
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

        self.Qsa = {}   # (s, a) -> mean action value
        self.Nsa = {}   # (s, a) -> visit count
        self.Ns = {}    # s -> visit count
        self.Ps = {}    # s -> policy vector from the network
        self.Es = {}    # s -> cached game-ended value (from perspective of player to move)
        self.Vs = {}    # s -> valid-moves mask

    def get_action_probs(self, canonical_board, temp=1.0, add_noise=False):
        """
        Run `num_sims` simulations from `canonical_board` and return the
        resulting visit-count policy (a length-action_size numpy array).
        `add_noise=True` mixes Dirichlet noise into the ROOT policy, exactly
        as AlphaZero does during self-play, to keep exploration alive.
        """
        s = self.game.string_rep(canonical_board)

        for i in range(self.num_sims):
            self._search(canonical_board, root=(i == 0 and add_noise))

        counts = np.array([self.Nsa.get((s, a), 0) for a in range(self.game.action_size)],
                           dtype=np.float64)

        if temp == 0:
            best = np.argwhere(counts == counts.max()).flatten()
            probs = np.zeros_like(counts)
            probs[np.random.choice(best)] = 1.0
            return probs

        counts = counts ** (1.0 / temp)
        total = counts.sum()
        if total == 0:
            # Shouldn't normally happen, fall back to uniform over valid moves
            valid = self.game.get_valid_moves(canonical_board)
            return valid / valid.sum()
        return counts / total

    def _search(self, board, root=False):
        s = self.game.string_rep(board)

        # --- terminal check (cached) ---
        if s not in self.Es:
            self.Es[s] = self.game.get_game_ended(board, 1)
        if self.Es[s] != 0:
            return -self.Es[s]  # value from the PARENT's perspective

        # --- leaf expansion ---
        if s not in self.Ps:
            policy, value = self.net.predict(board)
            valid = self.game.get_valid_moves(board)

            policy = policy * valid
            total = policy.sum()
            if total > 0:
                policy /= total
            else:
                # network assigned ~0 to all valid moves; fall back to uniform
                policy = valid / valid.sum()

            self.Ps[s] = policy
            self.Vs[s] = valid
            self.Ns[s] = 0
            return -value

        valid = self.Vs[s]
        policy = self.Ps[s]

        if root and self.dirichlet_eps > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(policy))
            policy = (1 - self.dirichlet_eps) * policy + self.dirichlet_eps * noise
            policy = policy * valid
            policy /= policy.sum()

        # --- select the action maximizing PUCT ---
        best_score, best_a = -float("inf"), -1
        sqrt_ns = math.sqrt(self.Ns[s] + 1e-8)
        for a in range(self.game.action_size):
            if not valid[a]:
                continue
            q = self.Qsa.get((s, a), 0.0)
            n = self.Nsa.get((s, a), 0)
            u = self.c_puct * policy[a] * sqrt_ns / (1 + n)
            score = q + u
            if score > best_score:
                best_score, best_a = score, a

        a = best_a
        next_board = self.game.get_next_state(board, 1, a)
        next_board = self.game.get_canonical_form(next_board, -1)

        value = self._search(next_board)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + value) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = value
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -value
