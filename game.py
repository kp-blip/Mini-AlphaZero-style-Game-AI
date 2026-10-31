"""
game.py
-------
A minimal Tic-Tac-Toe environment written in the "AlphaZero" style:
every board is represented as a length-9 numpy array of {-1, 0, 1}
(0 = empty, 1 = the piece of "player 1", -1 = the piece of "player -1").

Only 5 functions are required by the MCTS / self-play code:
    get_init_board()
    get_valid_moves(board)
    get_next_state(board, player, action)
    get_game_ended(board, player)
    get_canonical_form(board, player)

Swap this file out (keeping the same interface) to plug in Connect-4,
Gomoku, or any other small perfect-information 2-player game.
"""

import numpy as np

N = 3  # board is N x N


class TicTacToe:
    action_size = N * N
    board_size = N * N

    # --- board construction ---
    def get_init_board(self):
        return np.zeros(self.board_size, dtype=np.int8)

    # --- moves ---
    def get_valid_moves(self, board):
        """Boolean mask of length action_size, True where a move is legal."""
        return board == 0

    def get_next_state(self, board, player, action):
        """Return a NEW board with `player`'s piece placed at `action`."""
        b = board.copy()
        b[action] = player
        return b

    # --- win / draw checking ---
    def _lines(self):
        idx = np.arange(self.board_size).reshape(N, N)
        lines = []
        lines.extend(idx[i, :].tolist() for i in range(N))       # rows
        lines.extend(idx[:, j].tolist() for j in range(N))       # cols
        lines.append([idx[i, i] for i in range(N)])              # diag
        lines.append([idx[i, N - 1 - i] for i in range(N)])      # anti-diag
        return lines

    def _has_won(self, board, player):
        for line in self._lines():
            if all(board[i] == player for i in line):
                return True
        return False

    def get_game_ended(self, board, player):
        """
        Returns, from the perspective of `player`:
            0     -> game not over
            1     -> `player` has won
           -1     -> `player` has lost
            1e-4  -> draw (small nonzero so callers can distinguish
                           "draw" from "not over" with a simple `if`)
        """
        if self._has_won(board, player):
            return 1
        if self._has_won(board, -player):
            return -1
        if not np.any(board == 0):
            return 1e-4  # draw
        return 0

    # --- canonicalisation ---
    def get_canonical_form(self, board, player):
        """
        The network always sees the board from the CURRENT player's point
        of view: multiplying by `player` swaps whose stones look like "+1".
        """
        return board * player

    # --- symmetries (used to augment self-play data) ---
    def get_symmetries(self, board, pi):
        """
        Tic-Tac-Toe has 8 board symmetries (4 rotations x 2 reflections).
        Returns a list of (board, pi) pairs including the identity.
        """
        b2d = board.reshape(N, N)
        p2d = np.array(pi).reshape(N, N)
        out = []
        for k in range(4):
            rb = np.rot90(b2d, k)
            rp = np.rot90(p2d, k)
            out.append((rb.flatten(), rp.flatten().tolist()))
            out.append((np.fliplr(rb).flatten(), np.fliplr(rp).flatten().tolist()))
        return out

    # --- misc ---
    def string_rep(self, board):
        return board.tobytes()

    def display(self, board):
        symbols = {1: "X", -1: "O", 0: "."}
        rows = []
        for r in range(N):
            rows.append(" ".join(symbols[int(x)] for x in board[r * N:(r + 1) * N]))
        return "\n".join(rows)
