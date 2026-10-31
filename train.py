"""
train.py
--------
The full AlphaZero training loop, scaled down to run on a laptop CPU in
minutes instead of on a data-center in days:

    for iteration in range(num_iterations):
        1) SELF-PLAY: play `games_per_iter` games with the CURRENT best
           network guiding MCTS, collect (board, pi, z) examples.
        2) TRAIN: take a fresh copy of the network and train it on a
           batch sampled from the replay buffer.
        3) ARENA: play the newly trained network against the current
           best network. Only replace "best" if the challenger wins
           a large enough fraction of games (this is what AlphaZero /
           AlphaGo Zero do to avoid regressions from a bad update).

Run:
    python train.py                      # sensible defaults
    python train.py --iterations 20      # train longer
    python train.py --sims 200           # stronger (slower) MCTS

Checkpoints are written to ./checkpoints/best.npz after every accepted
update, so you can stop training at any time and still have a usable model.
"""

import argparse
import copy
import os
import time
from collections import deque

import numpy as np

from game import TicTacToe
from network import PolicyValueNet
from self_play import execute_self_play_game
from evaluate import mcts_player, random_player, MinimaxPlayer, play_match


def train(
    iterations=15,
    games_per_iter=25,
    sims=60,
    arena_sims=60,
    arena_games=20,
    win_frac_to_accept=0.55,
    epochs=5,
    batch_size=64,
    lr=1e-3,
    replay_buffer_size=20000,
    checkpoint_dir="checkpoints",
    seed=0,
    verbose=True,
):
    np.random.seed(seed)
    os.makedirs(checkpoint_dir, exist_ok=True)
    game = TicTacToe()

    best_net = PolicyValueNet(game.board_size, game.action_size, seed=seed)
    buffer = deque(maxlen=replay_buffer_size)

    minimax = MinimaxPlayer(game)

    for it in range(1, iterations + 1):
        t0 = time.time()

        # --- 1) self-play with the current best network ---
        new_examples = []
        for _ in range(games_per_iter):
            new_examples.extend(execute_self_play_game(game, best_net, num_sims=sims))
        buffer.extend(new_examples)

        # --- 2) train a challenger on the replay buffer ---
        # Deep-copy so training the challenger never mutates best_net in place;
        # if the arena rejects it, best_net is untouched.
        challenger = copy.deepcopy(best_net)

        data = list(buffer)
        p_loss = v_loss = 0.0
        for ep in range(epochs):
            np.random.shuffle(data)
            for start in range(0, len(data), batch_size):
                chunk = data[start:start + batch_size]
                boards = np.array([c[0] for c in chunk])
                pis = np.array([c[1] for c in chunk])
                zs = np.array([c[2] for c in chunk])
                p_loss, v_loss = challenger.train_step(boards, pis, zs, lr=lr)

        # --- 3) arena: challenger vs current best ---
        p1 = mcts_player(game, challenger, num_sims=arena_sims)
        p2 = mcts_player(game, best_net, num_sims=arena_sims)
        c_wins, b_wins, draws = play_match(game, p1, p2, num_games=arena_games)
        total_decisive = c_wins + b_wins
        win_frac = c_wins / total_decisive if total_decisive > 0 else 0.5

        accepted = (total_decisive == 0) or (win_frac >= win_frac_to_accept)
        if accepted:
            best_net = challenger
            best_net.save(os.path.join(checkpoint_dir, "best.npz"))

        if verbose:
            dt = time.time() - t0
            status = "ACCEPTED" if accepted else "rejected"
            print(f"[iter {it:3d}/{iterations}] buffer={len(buffer):5d} "
                  f"p_loss={p_loss:.3f} v_loss={v_loss:.3f}  "
                  f"arena: challenger {c_wins}-{b_wins}-{draws} best  -> {status}  "
                  f"({dt:.1f}s)")

    # --- final sanity check vs a perfect Minimax player ---
    if verbose:
        print("\nEvaluating final network against PERFECT play (Minimax)...")
    mm_play = lambda b: minimax.play(b)
    agent_play = mcts_player(game, best_net, num_sims=max(100, sims))
    wins, losses, draws = play_match(game, agent_play, mm_play, num_games=20)
    if verbose:
        print(f"vs Minimax: {wins} wins, {losses} losses, {draws} draws "
              f"(a perfect agent draws every game and never loses)")

    rand_play = random_player(game)
    wins, losses, draws = play_match(game, agent_play, rand_play, num_games=20)
    if verbose:
        print(f"vs Random : {wins} wins, {losses} losses, {draws} draws")

    return best_net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Mini-AlphaZero Tic-Tac-Toe agent")
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--games-per-iter", type=int, default=25)
    parser.add_argument("--sims", type=int, default=60, help="MCTS simulations during self-play")
    parser.add_argument("--arena-sims", type=int, default=60, help="MCTS simulations during arena games")
    parser.add_argument("--arena-games", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(
        iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        sims=args.sims,
        arena_sims=args.arena_sims,
        arena_games=args.arena_games,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
