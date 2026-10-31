"""
play.py
-------
Play interactive Tic-Tac-Toe against the trained agent in the terminal.

Run:
    python play.py                          # loads checkpoints/best.npz
    python play.py --checkpoint some.npz
    python play.py --sims 200                # let the AI "think" harder
    python play.py --human-second            # let the AI move first
"""

import argparse
import os

import numpy as np

from game import TicTacToe
from network import PolicyValueNet
from mcts import MCTS


def print_board(game, board):
    print()
    print(game.display(board))
    print()


def get_human_move(game, board):
    valid = np.flatnonzero(game.get_valid_moves(board))
    valid_list = [int(x) for x in valid]
    while True:
        raw = input(f"Your move {valid_list} (row,col or 0-8): ").strip()
        try:
            if "," in raw:
                r, c = (int(x) for x in raw.split(","))
                a = r * 3 + c
            else:
                a = int(raw)
        except ValueError:
            print("Could not parse that, try again.")
            continue
        if a in valid_list:
            return a
        print("Illegal move, try again.")


def main():
    parser = argparse.ArgumentParser(description="Play Tic-Tac-Toe against the trained agent")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.npz")
    parser.add_argument("--sims", type=int, default=150)
    parser.add_argument("--human-second", action="store_true",
                         help="Let the AI make the first move instead of you")
    args = parser.parse_args()

    game = TicTacToe()
    net = PolicyValueNet(game.board_size, game.action_size)

    if os.path.exists(args.checkpoint):
        net.load(args.checkpoint)
        print(f"Loaded trained weights from {args.checkpoint}")
    else:
        print(f"WARNING: no checkpoint found at {args.checkpoint} -- "
              f"the AI will play with UNTRAINED (near-random) weights.\n"
              f"Run `python train.py` first to actually train it.")

    board = game.get_init_board()
    human_player = -1 if args.human_second else 1
    cur_player = 1

    print("You are 'X' if you move first, 'O' if the AI moves first.")
    print("Board positions are numbered 0-8, left-to-right, top-to-bottom:")
    print(" 0 1 2\n 3 4 5\n 6 7 8\n")

    while True:
        print_board(game, board)
        canonical = game.get_canonical_form(board, cur_player)

        if cur_player == human_player:
            action = get_human_move(game, canonical)
        else:
            mcts = MCTS(game, net, num_sims=args.sims, dirichlet_eps=0.0)
            pi = mcts.get_action_probs(canonical, temp=0)
            action = int(np.argmax(pi))
            print(f"AI plays {action}")

        board = game.get_next_state(board, cur_player, action)
        ended = game.get_game_ended(board, cur_player)

        if ended != 0:
            print_board(game, board)
            if ended == 1e-4:
                print("It's a draw!")
            elif cur_player == human_player:
                print("You win!")
            else:
                print("The AI wins!")
            break

        cur_player = -cur_player


if __name__ == "__main__":
    main()
