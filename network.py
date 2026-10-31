"""
network.py
----------
A tiny policy+value network, implemented from scratch with NumPy so the
whole project runs with zero heavyweight dependencies (no PyTorch/TensorFlow
needed). This is the "f_theta(s) -> (p, v)" function from the AlphaZero
paper: given a board, predict a move-probability distribution `p` and a
scalar value estimate `v` in [-1, 1] of how good the position is for the
player to move.

Architecture:
    input (board_size)
      -> Dense(hidden) -> ReLU
      -> Dense(hidden) -> ReLU
      -> policy head: Dense(action_size) -> softmax
      -> value head : Dense(1) -> tanh

Training uses full manual backprop + the Adam optimizer (also implemented
by hand below). For a game as small as Tic-Tac-Toe this trains in seconds
on a CPU.
"""

import numpy as np


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


class AdamParam:
    """Wraps one weight/bias array with its own Adam moment estimates."""

    def __init__(self, value):
        self.value = value
        self.m = np.zeros_like(value)
        self.v = np.zeros_like(value)

    def update(self, grad, lr, beta1, beta2, eps, t):
        self.m = beta1 * self.m + (1 - beta1) * grad
        self.v = beta2 * self.v + (1 - beta2) * (grad ** 2)
        m_hat = self.m / (1 - beta1 ** t)
        v_hat = self.v / (1 - beta2 ** t)
        self.value -= lr * m_hat / (np.sqrt(v_hat) + eps)


class PolicyValueNet:
    def __init__(self, board_size, action_size, hidden=64, seed=0):
        rng = np.random.default_rng(seed)
        self.board_size = board_size
        self.action_size = action_size
        self.hidden = hidden

        def he(fan_in, fan_out):
            return rng.normal(0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))

        self.W1 = AdamParam(he(board_size, hidden))
        self.b1 = AdamParam(np.zeros(hidden))
        self.W2 = AdamParam(he(hidden, hidden))
        self.b2 = AdamParam(np.zeros(hidden))
        self.Wp = AdamParam(he(hidden, action_size))
        self.bp = AdamParam(np.zeros(action_size))
        self.Wv = AdamParam(he(hidden, 1))
        self.bv = AdamParam(np.zeros(1))

        self._params = [self.W1, self.b1, self.W2, self.b2,
                         self.Wp, self.bp, self.Wv, self.bv]
        self._t = 0  # Adam timestep

    # --- forward ---
    def forward(self, X):
        """X: (N, board_size) -> policy (N, action_size), value (N, 1)"""
        z1 = X @ self.W1.value + self.b1.value
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.W2.value + self.b2.value
        a2 = np.maximum(z2, 0)

        policy_logits = a2 @ self.Wp.value + self.bp.value
        policy = softmax(policy_logits)

        value_pre = a2 @ self.Wv.value + self.bv.value
        value = np.tanh(value_pre)

        cache = (X, z1, a1, z2, a2, value_pre, value)
        return policy, value, cache

    def predict(self, board):
        """Single-board convenience wrapper used by MCTS."""
        X = board.astype(np.float64).reshape(1, -1)
        policy, value, _ = self.forward(X)
        return policy[0], float(value[0, 0])

    # --- backward ---
    def train_step(self, boards, pis, zs, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        One gradient step over a batch.
        boards: (N, board_size)
        pis:    (N, action_size)  MCTS visit-count distributions (targets)
        zs:     (N,) or (N,1)     game outcomes in [-1, 1] (targets)
        Returns (policy_loss, value_loss) for logging.
        """
        X = np.asarray(boards, dtype=np.float64)
        pis = np.asarray(pis, dtype=np.float64)
        zs = np.asarray(zs, dtype=np.float64).reshape(-1, 1)
        N = X.shape[0]

        policy, value, cache = self.forward(X)
        X, z1, a1, z2, a2, value_pre, value = cache

        # --- losses (for logging only) ---
        policy_loss = -np.sum(pis * np.log(policy + 1e-12)) / N
        value_loss = np.mean((zs - value) ** 2)

        # --- output-layer gradients ---
        d_policy_logits = (policy - pis) / N                      # softmax + CE
        d_value_pre = 2.0 * (value - zs) * (1 - value ** 2) / N   # tanh + MSE

        dWp = a2.T @ d_policy_logits
        dbp = d_policy_logits.sum(axis=0)
        dWv = a2.T @ d_value_pre
        dbv = d_value_pre.sum(axis=0)

        da2 = d_policy_logits @ self.Wp.value.T + d_value_pre @ self.Wv.value.T
        dz2 = da2 * (z2 > 0)

        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.W2.value.T
        dz1 = da1 * (z1 > 0)

        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        # ---- Adam update ----
        self._t += 1
        grads = [dW1, db1, dW2, db2, dWp, dbp, dWv, dbv]
        for p, g in zip(self._params, grads):
            p.update(g, lr, beta1, beta2, eps, self._t)

        return policy_loss, value_loss

    # --- persistence ---
    def save(self, path):
        np.savez(path, **{f"p{i}": p.value for i, p in enumerate(self._params)})

    def load(self, path):
        data = np.load(path)
        for i, p in enumerate(self._params):
            p.value = data[f"p{i}"]
