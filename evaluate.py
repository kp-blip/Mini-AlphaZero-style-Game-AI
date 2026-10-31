"""
evaluate.py
-----------
Two evaluation tools:

1. `MinimaxPlayer` - plays PERFECT Tic-Tac-Toe (exhaustive minimax with
   memoization; the state space is tiny so this is instant). A trained
   agent that never loses to this player has essentially "solved" the game.

2. `play_match` / `arena` - runs N games between any two "players"
   (functions board -> action) and reports the win/draw/loss record.
   Used both to sanity-check against Minimax and to compare a newly
   trained network against the previous best one before accepting it.
"""

import numpy as np
from functools import lru_cache


class MinimaxPlayer:
    """Optimal Tic-Tac-Toe player. Never loses."""

    def __init__(self, game):
        self.game = game

    def _minimax(self, board_tuple, player):
        board = np.array(board_tuple, dtype=np.int8)
        ended = self.game.get_game_ended(board, player)
        if ended != 0:
            return ended, None

        best_val, best_action = -2, None
        for a in np.flatnonzero(self.game.get_valid_moves(board)):
            nb = self.game.get_next_state(board, player, a)
            val, _ = self._minimax(tuple(nb.tolist()), -player)
            val = -val
            if val > best_val:
                best_val, best_action = val, a
        return best_val, best_action

    def play(self, canonical_board):
        """canonical_board is already from the perspective of `player`=1."""
        _, action = self._minimax(tuple(canonical_board.tolist()), 1)
        return action


def random_player(game):
    def play(canonical_board):
        valid = np.flatnonzero(game.get_valid_moves(canonical_board))
        return int(np.random.choice(valid))
    return play


def mcts_player(game, net, num_sims=100, c_puct=1.5):
    """Wraps a trained network + fresh MCTS into a `play(board)` function."""
    from mcts import MCTS

    def play(canonical_board):
        mcts = MCTS(game, net, c_puct=c_puct, num_sims=num_sims, dirichlet_eps=0.0)
        pi = mcts.get_action_probs(canonical_board, temp=0)
        return int(np.argmax(pi))
    return play


def play_match(game, player1, player2, num_games=20):
    """
    player1 / player2: functions (canonical_board) -> action.
    Returns (p1_wins, p2_wins, draws). Who moves first alternates each
    game so neither player benefits from a first-move advantage.
    """
    p1_wins = p2_wins = draws = 0

    for g in range(num_games):
        board = game.get_init_board()
        players = {1: player1, -1: player2} if g % 2 == 0 else {1: player2, -1: player1}
        cur_player = 1

        while True:
            canonical = game.get_canonical_form(board, cur_player)
            action = players[cur_player](canonical)
            board = game.get_next_state(board, cur_player, action)
            ended = game.get_game_ended(board, cur_player)
            if ended != 0:
                if ended == 1e-4:
                    draws += 1
                else:
                    winner_fn = players[cur_player] if ended == 1 else players[-cur_player]
                    if winner_fn is player1:
                        p1_wins += 1
                    else:
                        p2_wins += 1
                break
            cur_player = -cur_player

    return p1_wins, p2_wins, draws
