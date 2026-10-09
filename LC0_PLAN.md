# Opponent Modeling in Lc0: Plan
This document describes how opponent modeling would be added to Lc0, and why. It is the counterpart of [PLAN.md](PLAN.md), which covers the Stockfish version.

File and line references point to the unmodified Lc0 v0.32.1 source, as committed in `69f1164` under [lc0-src/](lc0-src/). Line numbers will drift once changes are made.

## Why Lc0
The Stockfish version models the opponent as playing the best move, for it, out of a restricted set of moves. That is still minimax, so the opponent is weak at deciding which moves are acceptable but perfect at choosing among them. In testing, this made the modeled opponent avoid recapturing in the Belgrade Gambit, because it could foresee its own later mistakes.

The natural fix is an expected value: score an opponent position as the probability-weighted average over the moves it is likely to play. Inside Stockfish's alpha-beta search this proved too expensive. Each opponent node needed several searches with different windows, the hash table could not share their results, and the cost multiplied with every opponent move in a line.

Lc0 searches with Monte Carlo tree search (MCTS), which works with averages. Two measurements also favor it:
- **Maia predicts human moves better.** On the same 20,736 positions from Lichess games, Maia's top move matched the played move 48.8% of the time, against 37.2% for the Stockfish version's low-depth search. Maia was better in every rating band (see [MEASUREMENTS.md](MEASUREMENTS.md)).
- **Maia runs on Lc0.** The Maia networks are published in Lc0's network format, and Lc0 runs them on this computer's GPU.

## The Key Observation
In Lc0, a node's value Q is the running average of every evaluation backed up through it. `Node::FinalizeScoreUpdate` ([node.cc:356-363](lc0-src/src/search/classic/node.cc#L356-L363)) adds each new value `v` as `wl_ += multivisit * (v - wl_) / (n_ + multivisit)`. So a node's Q is the average of its children's Q values, weighted by how many visits each child received.

Normally, Lc0's selection rule (PUCT) sends most visits to the best-looking child, so Q tends towards the best child's value: in effect, minimax. If the visits at an opponent node are instead shared out in proportion to the opponent's move probabilities, the same averaging gives:

> Q(opponent node) ≈ Σ p(move) × Q(child after that move)

That is exactly the expected value over the opponent's likely moves. The backup code doesn't need to change at all; only how visits are distributed at opponent nodes.

## Design
### Who is the opponent
As in the Stockfish version, the opponent is the side not to move at the root of each search. Nodes where the opponent is to move are opponent nodes. All other nodes are searched exactly as Lc0 normally does.

### Two networks
- **The main network** evaluates every position: its value is backed up as usual, and its policy guides the search at our nodes.
- **A Maia network**, chosen for the opponent's rating, provides the move probabilities at opponent nodes. Its value is not used.

When an opponent node is expanded, both networks evaluate it. Lc0 currently sends each batch of positions to one backend ([search.cc:1164](lc0-src/src/search/classic/search.cc#L1164), [search.cc:1460](lc0-src/src/search/classic/search.cc#L1460)). The opponent-node positions in a batch would also be sent to a second backend running Maia. When the results come back, the main network's value is used as usual. The opponent node's move probabilities are taken from Maia instead of the main network ([search.cc:2175](lc0-src/src/search/classic/search.cc#L2175), where each edge's probability is set).

Maia networks are small (6 blocks, 64 filters), so the extra evaluation should cost much less than the main network's.

### Sharing out visits at opponent nodes
Node selection happens in `SearchWorker::PickNodesToExtendTask` ([search.cc:1564-1911](lc0-src/src/search/classic/search.cc#L1564-L1911)). At each node it scores every child with PUCT ([search.cc:1733-1734](lc0-src/src/search/classic/search.cc#L1733-L1734)):

```
score = P × cpuct × √N / (1 + n_started) + Q
```

and sends the next visits to the child with the highest score. It also estimates how many visits the best child can take before the second best would overtake it, and sends that many at once ([search.cc:1777-1787](lc0-src/src/search/classic/search.cc#L1777-L1787)).

At opponent nodes, the score becomes the child's share of visits minus the visits it already has:

```
score = P_maia × (N + 1) − n_started
```

The child that is furthest below its share gets the next visit, so the visits track Maia's probabilities, and Q tracks the expected value. Q plays no part in the choice: the opponent is modeled as playing like Maia, not as looking for the best reply. The "visits until the second best overtakes" estimate follows from the same formula.

### Proven wins and losses
Lc0 marks positions as proven wins, losses or draws and propagates them up the tree ("sticky endgames", [search.cc:2285](lc0-src/src/search/classic/search.cc#L2285)). That propagation is minimax: if any child is a proven win for the side to move, the node is a proven win. At an opponent node that is no longer true, because the opponent plays its winning move only with some probability. So propagation would skip opponent nodes: a proven result there is backed up only through the averaging, like any other value. The simplest first version turns the `StickyEndgames` option off whenever opponent modeling is on.

### Options
- `OpponentModelWeights`: the Maia network file. Empty turns opponent modeling off, so Lc0 behaves exactly as normal.
- `OpponentModelTemperature`: a temperature for Maia's probabilities, 1.0 by default. Above 1 flattens them, so the opponent is less predictable; below 1 sharpens them.
- `OpponentModelRating` and `OpponentModelOwnRating` (added with Maia-3, see below): the modeled opponent's rating, and the rating of the side the engine plays, both 1300 by default. Maia-3 takes both as inputs. Maia-1 networks ignore them, since each is trained for one rating.
- `OpponentRootCandidates` (added later, see "Choosing the move at the root"): the number of root moves compared by sequential halving, 8 by default. 0 or 1 uses Lc0's usual rule.

### What stays the same
- Our nodes, including the root: PUCT with the main network, as usual.
- Choosing the move to play: by visits at the root, as usual.
- Time management, tree reuse between moves, and the search's multithreading.

## Expected Behavior
- **The Belgrade Gambit line** from the Stockfish testing: after `8.Nxe4`, Maia gives `...fxe4` a high probability, so most visits at that node go to the recapture and the engine has to prepare for it. Castling contributes only its probability's share.
- **Gambits and traps:** a move that sets a trap is scored by how often Maia falls into it, weighted against what happens when it doesn't. A trap that works 30% of the time, but leaves us worse otherwise, is scored accordingly, and is not assumed to work.
- **Scores are expected values against the modeled opponent,** not objective evaluations, so they will differ from plain Lc0's.

## Testing
1. **No model, no change:** with `OpponentModelWeights` empty, the search must behave exactly as unmodified Lc0. With a fixed node count and one thread, the chosen moves and node counts must match.
2. **Visit shares:** at the root's children (opponent nodes), with verbose move statistics, the visit shares must track Maia's probabilities.
3. **Chosen positions:** the Belgrade Gambit line, `1.e4 c5`, and positions with a single hard-to-find defense, compared with plain Lc0 and with the Stockfish version.
4. **Games:** matches against Maia at a set rating, against plain Lc0 at equal nodes, and against the Stockfish version, using `fastchess` or `cutechess-cli`.

## Implementation Status
Implemented as planned, in [search.h](lc0-src/src/search/classic/search.h), [search.cc](lc0-src/src/search/classic/search.cc) and [wrapper.cc](lc0-src/src/search/classic/wrapper.cc):

- **The opponent model's backend** is created by `ClassicSearch::UpdateOpponentModel` when `OpponentModelWeights` or `OpponentModelTemperature` changes. It uses an options dictionary whose parent is the engine's options, with the weights file and policy temperature replaced, and gets its own evaluation cache. The tree is cleared when the model or the opponent's color changes, because its values and opponent-node probabilities depend on both.
- **Each search worker** creates a second computation for the opponent model. At opponent nodes, `ProcessPickedTask` sends the position to both networks. A node is only evaluated out of order when both results came from cache, and `RunNNComputation` runs both. `FetchSingleNodeResult` takes the opponent node's move probabilities from the opponent model.
- **Selection** at opponent nodes gives the next visit to the move furthest below its share, `P × (visits + 1) − visits started`. The number of visits it can take before the second-furthest move catches up is `lead / (1 − P(best) + P(second)) + 1`.
- **Proven results** are not propagated while the opponent model is on.
- Without `OpponentModelWeights`, every new branch is skipped, so the search runs Lc0's original code.

### Maia-3
[Maia-3](https://github.com/CSSLab/maia3) predicts real moves better than Maia-1 (see [MEASUREMENTS.md](MEASUREMENTS.md)) and covers ratings 600 to 3000 in one network, but it is a PyTorch transformer with its own inputs and move encoding, so Lc0's backends cannot run it. It gets its own backend, [network_maia3.cc](lc0-src/src/neural/backends/network_maia3.cc), used when `OpponentModelWeights` names a `.onnx` file:

- **Export.** [tools/export_maia3.py](tools/export_maia3.py) exports a checkpoint to ONNX, in fp32 or fp16, and checks it against the original PyTorch model. Its forward pass is Maia-3's own, except that the 256 promotion logits are computed in one operation instead of a Python loop of 256 small ones. The result matches the original to within 3e-5 in the logits.
- **Encoding in C++.** Lc0's board is already in the frame Maia-3 expects (the side to move's, ranks flipped for Black), so each position's 12 piece planes are read straight from its bitboards, for the last 8 positions, oldest first, padded at the front with the oldest. Moves map to Maia-3's vocabulary as `from × 64 + to`, castling as the king's two-square move, and promotions at `4096 + (from file × 8 + to file) × 4 + piece`. This replaces the Python encoding, which managed about 2,000 positions a second.
- **Running.** ONNX Runtime's CUDA provider runs the network. Maia-3 is many small GPU operations, so a run cost about 12 ms whatever the batch size, which held the search to about 10,000 nodes per second. Now each of the batch sizes 16, 32, 64, 128 and 256 gets its own session with fixed shapes and GPU buffers, and ONNX Runtime records each run once as a CUDA graph and replays it. A batch runs at the smallest size that holds it. The graphs are recorded when the model loads, because recording blocks CUDA calls from other threads, such as the main network's.
- **Probabilities and value.** The policy is a softmax over the legal moves' logits, at `OpponentModelTemperature`. The value (loss, draw, win) is filled in but not used.

 In the Belgrade Gambit line, after `8.Nxe4`, Black's replies got 94.2%, 4.6% and 0.3% of the visits, against Maia-1300 probabilities of 94.17%, 4.59% and 0.29%.
- **No model, no change.** Unmodified Lc0 is not exactly reproducible between runs, even with one thread and no task workers: two runs of it differed on `1.e4 c5`. Compared at 20,000 nodes, the final root statistics of the modified and unmodified builds were identical in two positions out of four, and in the other two the differences were within the unmodified build's own run-to-run variation.
- **Speed.** About 62,000 nodes per second with Maia, against about 140,000 for plain Lc0 with the release's network on the RTX 4080: about 2.2 times slower, because opponent nodes need two evaluations.
- **Maia-3 matches the reference.** Lc0's probabilities at opponent nodes match StockfishTrapFinder's `Maia3Net` (PyTorch): after `8.Nxe4` in the Belgrade Gambit line, with ratings 1100 against 1900, 92.55%, 3.79% and 1.07% from Lc0 against 92.54%, 3.79% and 1.07%, with the fp32 23M network. The fp16 networks differ by about 0.1 percentage points. The visit shares follow the probabilities, as with Maia-1, and the White-to-move frame checks out too.
- **Maia-3 speed**, nodes per second after `1.e4 c5`, with the model loaded, on the RTX 4080:

  | Opponent model | Nodes per second |
  |---|---|
  | None | 156,000 |
  | Maia-1 (1300) | 69,000 |
  | Maia3-5M, fp16 | 38,000 |
  | Maia3-23M, fp16 | 24,000 |
  | Maia3-23M, fp32 | about 18,000 |
  | Maia3-79M, fp16 | 10,000 |

  Before CUDA graphs and the vectorized promotions, Maia3-23M fp16 ran at about 10,000. The networks are now limited by their computation: at batch 256, the 23M network takes about 80 microseconds per position.
- **Different choices.** With Maia-1300, the engine plays `2.d4` after `1.e4 c5` (plain Lc0: `2.Nf3`), and `Qc3` instead of `Bxd6` in the `Bf4` line from the Stockfish testing.
- **Scores are expected values,** so they are far from plain Lc0's: for example +1.92 from the start position against Maia-1300, against +0.28. A strong engine playing a 1300 player is expected to win from most positions. What matters for choosing a move is how the candidates compare.

### Choosing the move at the root
**The problem.** With the opponent model, a move's value tends to rise the more it is searched, because a deeper search finds more of the opponent's likely mistakes. Lc0's usual rule (PUCT) gives most visits to the move that currently looks best, and plays the move with most visits. So whichever move takes the lead early gets more visits, looks better for it, and keeps the lead. From the start position, `1.Nf3` took 127,000 of 135,000 visits and rose from +0.10 to +0.63. Its rivals stayed near their first estimates. Given 51,000 nodes each with `searchmoves`, the moves scored:

| Move | Score |
|---|---|
| `1.e4` | +2.47 |
| `1.Nf3` | +1.76 |
| `1.d4` | +1.70 |
| `1.c4` | +1.69 |

Raising the root's exploration (`RootHasOwnCpuctParams`, `CPuctAtRoot` 5 or 8) only changed which move won the race: one move still took 93% of the visits.

**Sequential halving.** With the opponent model on, the root uses sequential halving, as in Gumbel MuZero, instead:
- **Rounds.** The `OpponentRootCandidates` moves with the highest priors (8 by default) get equal visits. When all of them reach the round's target, they are ranked by value, the worse half is dropped, and the rest go on to the next round.
- **Runs.** A run spreads a budget of visits evenly over its rounds, `ceil(log2(candidates))` of them. When one move is left, a new run starts with twice the budget, so the search can go on for any time. The new run's targets are absolute visit counts, so moves already above them take no visits until the others catch up.
- **The best move** is the one ranked best at the last halving, so moves are only ever compared at equal visits. The reported score and principal variation follow it, unless a proven win or loss decides.
- **Stopping.** Lc0's smart pruning stops a search once the most-visited move can no longer be caught. Here, visits no longer decide the move, so the stoppers are given equal visit counts, which turns that check off. A search now runs to its full node or time limit. In timed games that uses more clock time per move.

The state lives in `Search::RootHalving` and is updated whenever the root is picked from, with `nodes_mutex_` held exclusively. The visits go to the candidate with the fewest visits started, up to the next fewest.

**Results**, 300,000 nodes:
- **From the start position,** the candidates finish within about 0.06 of each other: `1.e4` with 4 or 16 candidates, `1.d4` with 8. Against a 1300 player the first move matters little.
- **In the test game,** the engine finds the same moves as with PUCT: `6.e4` after `5...b6`, and `8.Bxa6` after `7...a6`. `Bxa6` wins the equal-visit comparison clearly, 0.961 against 0.912 for `Nc3`, at about 68,000 visits each.

## Open Questions
- **Which main network.** The Lc0 release ships with a network (`791556.pb.gz`) that is good enough to develop with. A stronger network, such as a BT4 transformer network, can come later.
- **Calibration.** A previous project reweighted Maia's probabilities with a low-depth Stockfish evaluation, moving part of a move's probability to the evaluation's best move by `max_rate × (1 − exp(−loss / scale_cp))`, to suppress blunders a quick look would catch. Whether that predicts real moves better can be measured offline with the measurement tool before building it in. Building it in would need a low-depth evaluation of every candidate at each new opponent node, either a Stockfish evaluator compiled into Lc0 (CPU) or the main network's evaluation of the candidate positions (GPU). **Resolved: not needed.** Measured on 7,451 positions from full blitz and rapid games between players rated 1200–1399, raw Maia3-23M at temperature 1.0 already matches how often players blunder, in the opening, the middlegame and late in the game. No calibration setting predicted moves better (see "Tuning the calibration" in [MEASUREMENTS.md](MEASUREMENTS.md)).
- **Ratings outside Maia's range.** Maia covers 1100 to 1900. Maia-3, now supported, covers 600 to 3000.
- **Which Maia-3 size.** The 23M network predicts practically as well as the 79M one at 2.4 times the speed. The 5M one is another 1.6 times faster and predicts a little worse (51.4% top 1 against 52.3%).
