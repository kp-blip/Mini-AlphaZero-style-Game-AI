# Mini AlphaZero — Tic-Tac-Toe

A small, dependency-light implementation of the **AlphaZero algorithm**
(self-play + Monte Carlo Tree Search + a policy/value neural network,
with no hand-crafted game knowledge beyond the rules). Scaled down to
Tic-Tac-Toe so it trains in a couple of minutes on a plain CPU — but the
architecture is the real thing, and the same code extends to Connect-4,
Gomoku, or other small perfect-information games.

**Only dependency: NumPy.** The neural network (forward + backward pass +
Adam optimizer) is implemented from scratch, so there's no PyTorch/TensorFlow
install required.

## How it works

1. **Self-play** (`self_play.py`): the current best network plays games
   against itself. At every move, `MCTS` (`mcts.py`) runs ~50-100
   simulations guided by the network's policy/value predictions (PUCT
   search) to produce a *stronger* move distribution than the raw network
   output. This improved distribution `pi`, plus the eventual game
   outcome `z`, become one training example.
2. **Training** (`network.py`): a fresh copy of the network is trained on
   a batch of recent self-play examples to predict `pi` (policy head) and
   `z` (value head).
3. **Arena gating** (`evaluate.py`, used inside `train.py`): the newly
   trained network only replaces the "best" network if it wins a large
   enough fraction of games against the current best — this prevents a
   bad training step from making the agent worse.
4. Repeat. Over iterations the network bootstraps itself from
   near-random play up to (for Tic-Tac-Toe) **perfect, unbeatable play**.

## Files

| File | Purpose |
|---|---|
| `game.py` | Tic-Tac-Toe rules/board representation (swap this for another game) |
| `network.py` | Policy+value MLP, pure NumPy, manual backprop + Adam |
| `mcts.py` | PUCT Monte Carlo Tree Search guided by the network |
| `self_play.py` | Generates training examples via self-play |
| `evaluate.py` | Perfect Minimax player + arena match code |
| `train.py` | Main training loop (self-play → train → arena → repeat) |
| `play.py` | Play against the trained agent in your terminal |
| `checkpoints/best.npz` | A network already trained for you (see below) |

## Quickstart

```bash
# Play against the network that's already trained and included:
python play.py

# Let the AI move first:
python play.py --human-second

# Train from scratch (overwrites checkpoints/best.npz):
python train.py

# Train longer / stronger:
python train.py --iterations 25 --sims 100 --arena-games 30
```

## Result of the included training run

The included checkpoint was trained for 15 iterations (25 self-play games
each) and evaluated with 20 games against each opponent:

```
vs Minimax (perfect play): 0 wins, 0 losses, 20 draws   <- never loses
vs Random play:            20 wins, 0 losses, 0 draws   <- never even draws
```

Since Tic-Tac-Toe is a solved, drawn game with perfect play, "draws every
game against Minimax" is the ceiling — the agent has essentially solved it
purely from self-play, with no rules beyond move legality and win/draw/loss
detection ever hard-coded into it.

## Extending this to a bigger game (e.g. Connect-4)

The whole project is written against one interface (`game.py`). To try a
new game:
1. Copy `game.py`, implement `get_init_board`, `get_valid_moves`,
   `get_next_state`, `get_game_ended`, `get_canonical_form`, and
   `string_rep` for the new game.
2. Update `board_size` / `action_size` when constructing `PolicyValueNet`.
3. Everything else (`mcts.py`, `self_play.py`, `train.py`, `play.py`)
   works unchanged.

For a bigger board you'll likely want a bigger network
(`PolicyValueNet(..., hidden=128)`), more MCTS simulations, and (for real
strength) a GPU + PyTorch/JAX version of `network.py` — this NumPy version
is intentionally kept small and dependency-free for learning/demo purposes.

## Key hyperparameters (`train.py --help`)

| Flag | Meaning | Default |
|---|---|---|
| `--iterations` | Outer self-play/train/arena loops | 15 |
| `--games-per-iter` | Self-play games per iteration | 25 |
| `--sims` | MCTS simulations per move during self-play | 60 |
| `--arena-sims` | MCTS simulations per move during arena games | 60 |
| `--arena-games` | Games played to decide if the challenger replaces best | 20 |
| `--epochs` | Training epochs over the buffer per iteration | 5 |
| `--lr` | Adam learning rate | 1e-3 |
