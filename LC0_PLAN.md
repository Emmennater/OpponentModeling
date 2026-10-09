# Opponent Modeling in Lc0: Plan
This document describes how opponent modeling would be added to Lc0, and why. It is the counterpart of [PLAN.md](PLAN.md), which covers the Stockfish version.

File and line references point to the unmodified Lc0 v0.32.1 source, as committed in `69f1164` under [lc0-src/](lc0-src/). Line numbers will drift once changes are made.

## Why Lc0
The Stockfish version models the opponent as playing the best move, for it, out of a restricted set of moves. That is still minimax, so the opponent is weak at deciding which moves are acceptable but perfect at choosing among them. In testing, this made the modeled opponent avoid recapturing in the Belgrade Gambit, because it could foresee its own later mistakes.

The natural fix is an expected value: score an opponent position as the probability-weighted average over the moves it is likely to play. Inside Stockfish's alpha-beta search this proved too expensive. Each opponent node needed several searches with different windows, the hash table could not share their results, and the cost multiplied with every opponent move in a line.

Lc0 searches with Monte Carlo tree search (MCTS), which works with averages. Two measurements also favor it:
- **Maia predicts human moves better.** On the same 20,736 positions from Lichess games, Maia's top move matched the played move 48.8% of the time, against 37.2% for the Stockfish version's low-depth search. Maia was better in every rating band (see the README).
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

## Open Questions
- **Which main network.** The Lc0 release ships with a network (`791556.pb.gz`) that is good enough to develop with. A stronger network, such as a BT4 transformer network, can come later.
- **Calibration.** Whether Maia's probabilities should be sharpened or flattened can be checked with the measurement tool's log-likelihood, by rating.
- **Ratings outside Maia's range.** Maia covers 1100 to 1900. Stronger opponents would need a different model, or the main network's own policy with a temperature.
