"""
self_play.py
------------
Plays one full game of the agent against itself using MCTS to pick every
move, and records (board, pi, player) at each step. Once the game ends,
each recorded position is labeled with z = +1/-1/0 depending on whether
that position's player eventually won, lost, or drew -- these become the
training targets for the value head.
"""

import numpy as np
from mcts import MCTS


def execute_self_play_game(game, net, num_sims=100, c_puct=1.5,
                            temp_threshold=6, use_symmetries=True):
    """Returns a list of (canonical_board, pi, z) training examples."""
    mcts = MCTS(game, net, c_puct=c_puct, num_sims=num_sims)

    board = game.get_init_board()
    player = 1
    move_count = 0
    trajectory = []  # (canonical_board, pi, player_to_move)

    while True:
        canonical = game.get_canonical_form(board, player)
        temp = 1.0 if move_count < temp_threshold else 0.0
        pi = mcts.get_action_probs(canonical, temp=temp, add_noise=(move_count == 0))

        if use_symmetries:
            for sym_board, sym_pi in game.get_symmetries(canonical, pi):
                trajectory.append((sym_board, sym_pi, player))
        else:
            trajectory.append((canonical, pi, player))

        action = np.random.choice(len(pi), p=pi)
        board = game.get_next_state(board, player, action)

        ended = game.get_game_ended(board, player)
        if ended != 0:
            # ended is from the perspective of the player who just moved
            examples = []
            for b, p, ply in trajectory:
                if ply == player:
                    z = ended
                else:
                    z = -ended
                examples.append((b, p, z))
            return examples

        player = -player
        move_count += 1
