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
Lc0 marks positions as proven wins, losses or draws and propagates them up the tree ("sticky endgames"). That propagation is minimax: if any child is a proven win for the side to move, the node is a proven win. At an opponent node that is no longer true, because the opponent plays its winning move only with some probability. The first version turned propagation off whenever the opponent model was on.

That cost the engine its endings. With no proven mates, every winning move in a won ending scores about 1.0, and the engine shuffled: from king and queen against king, it took 19 to 23 moves to mate where 8 were possible, or did not mate within 40. So propagation is back, with a rule at opponent nodes that the opponent model does not break. An opponent node's value is an average over its moves, so it lies between their worst and best bounds, and the node is proven only when all its moves are proven to the same result. Then it does not matter which one the opponent plays. A proven node's moves left are averaged by the opponent model's probabilities. Our nodes keep Lc0's own rule. Root halving also ranks moves by value plus Lc0's bonus for shorter wins, as the search's own move choice does, instead of by value alone. In a won ending that bonus is what tells the moves apart. With both changes, the engine mated in 9, 8 and 9 moves from three king-and-queen positions, in two runs each, against Stockfish's best defense at 20,000 nodes a move. Plain Lc0 took 6, 8 to 9, and 9 to 13 moves; the fastest mates were 7 to 9 moves.

### Options
- `OpponentModelWeights`: the Maia network file. Empty turns opponent modeling off, so Lc0 behaves exactly as normal.
- `OpponentModelTemperature`: a temperature for Maia's probabilities, 1.0 by default. Above 1 flattens them, so the opponent is less predictable; below 1 sharpens them.
- `OpponentModelRating` and `OpponentModelOwnRating` (added with Maia-3, see below): the modeled opponent's rating, and the rating of the side the engine plays, both 1300 by default. Maia-3 takes both as inputs. Maia-1 networks ignore them, since each is trained for one rating.
- `OpponentRootCandidates` (added later, see "Choosing the move at the root"): the number of root moves compared by sequential halving, 8 by default. 0 or 1 uses Lc0's usual rule.
- `OpponentRootScout` (added later, see "Choosing the move at the root"): the root visits made with Lc0's usual rule before sequential halving starts, 1000 by default.
- `OpponentTreeReuse` (added later, see "Choosing the move at the root"): whether a search keeps the previous search's tree, off by default.
- `OpponentBestLean` (added later, see "Sharp moves need more search"): how far the value of a position where the engine is to move leans from the average of the moves searched there toward the most searched move's value, 0 by default.
- `OpponentCaution` and `OpponentRefutationProbability` (added later, see "Caution: refutations that are plausible"): how far an opponent position's value leans toward the opponent's best plausible reply, 0 by default, and the probability from which a reply counts fully as plausible, 5% by default; less likely replies count in proportion.
- `OpponentDiscount` (added later, see "Sharp moves need more search"): how much less a result counts for each half-move until the game ends, 1 (off) by default.
- `OpponentEngineMix` (added later, see "Sharp moves need more search"): the share of the main network's move probabilities mixed into the opponent model's, 0 by default.
- `OpponentTactics` (added later, see "Sharp moves need more search"): how much of a quick tactical look the modeled opponent takes, 0 by default.
- `OpponentMinProbability` (added later, see "Sharp moves need more search"): opponent moves the model gives less than this probability are taken as never played, as in the Stockfish version's threshold. 0 by default, which keeps every move.

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
- **Proven results** are propagated with a different rule at opponent nodes: proven only when all their moves are proven to the same result (see "Proven wins and losses").
- Without `OpponentModelWeights`, every new branch is skipped, so the search runs Lc0's original code.

### Maia-3
[Maia-3](https://github.com/CSSLab/maia3) predicts real moves better than Maia-1 (see [MEASUREMENTS.md](MEASUREMENTS.md)) and covers ratings 600 to 3000 in one network, but it is a PyTorch transformer with its own inputs and move encoding, so Lc0's backends cannot run it. It gets its own backend, [network_maia3.cc](lc0-src/src/neural/backends/network_maia3.cc), used when `OpponentModelWeights` names a `.onnx` file:

- **Export.** [tools/export_maia3.py](tools/export_maia3.py) exports a checkpoint to ONNX, in fp32 or fp16, and checks it against the original PyTorch model. Its forward pass is Maia-3's own, except that the 256 promotion logits are computed in one operation instead of a Python loop of 256 small ones. The result matches the original to within 3e-5 in the logits.
- **Encoding in C++.** Lc0's board is already in the frame Maia-3 expects (the side to move's, ranks flipped for Black), so each position's 12 piece planes are read straight from its bitboards, for the last 8 positions, oldest first, padded at the front with the oldest. Moves map to Maia-3's vocabulary as `from × 64 + to`, castling as the king's two-square move, and promotions at `4096 + (from file × 8 + to file) × 4 + piece`. This replaces the Python encoding, which managed about 2,000 positions a second.
- **Running.** ONNX Runtime's CUDA provider runs the network. Maia-3 is many small GPU operations, so a run cost about 12 ms whatever the batch size, which held the search to about 10,000 nodes per second. Now each of the batch sizes 16, 32, 64, 128 and 256 gets its own session with fixed shapes and GPU buffers, and ONNX Runtime records each run once as a CUDA graph and replays it. A batch runs at the smallest size that holds it. The graphs are recorded when the model loads, because recording blocks CUDA calls from other threads, such as the main network's.
- **One CUDA stream.** The copies of a batch to and from the GPU, and the run, all go on one CUDA stream of our own, passed to ONNX Runtime as its `user_compute_stream`. A first version copied with plain `cudaMemcpy` while ONNX Runtime ran on a stream of its own. CUDA does not order those against each other, so a run could read a batch before it had fully arrived, or its results could be read before they were written. In a 1,000,000-node search, about half of all batches came out different from a rerun after a full synchronization, with logits off by up to 13. Nodes then got another position's probabilities, and the engine expected moves like `7...Be7??`, hanging a bishop, after `1.e4 e5 2.d4 exd4 3.Qxd4 Nc6 4.Qe3 d6 5.Nc3 Nf6 6.Bd2 Bg4 7.f3`. With one stream, no batch differed in 4,000, and the search runs faster. Short tests had matched the reference by luck: they hit good batches.
- **One thread.** ONNX Runtime keeps a session's CUDA graph per calling thread, and Lc0 starts new search threads for every search. So when the search threads ran Maia-3 themselves, every search recorded the graphs again partway through. Recording blocks CUDA calls from other threads. When the main network synchronized during a recording, it failed with "operation not permitted when stream is capturing" and Lc0 exited. In En Croissant's game report, which runs a 500 ms search for every position, that happened within a few positions, and the GUI then kept sending `stop` to the dead engine. Now one worker thread owns the sessions: it creates them, records the graphs once while warming them up, runs every batch, and releases the sessions. The search threads hand it their batches and wait. A replay of the report's commands over a 38-position game then ran three times without failure, at the same speed.
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
  | Maia3-5M, fp16 | 55,000 |
  | Maia3-23M, fp16 | 35,000 |
  | Maia3-23M, fp32 | 21,000 |
  | Maia3-79M, fp16 | 15,000 |

  Before CUDA graphs and the vectorized promotions, Maia3-23M fp16 ran at about 10,000, and 24,000 before the single stream. The networks are now limited by their computation: at batch 256, the 23M network takes about 80 microseconds per position.
- **Different choices.** With Maia-1300, the engine plays `2.d4` after `1.e4 c5` (plain Lc0: `2.Nf3`), and `Qc3` instead of `Bxd6` in the `Bf4` line from the Stockfish testing.
- **Scores are expected values,** so they are far from plain Lc0's: for example +1.92 from the start position against Maia-1300, against +0.28. A strong engine playing a 1300 player is expected to win from most positions. What matters for choosing a move is how the candidates compare.

### Choosing the move at the root
**The problem.** With the opponent model, a move's value tends to rise the more it is searched, because a deeper search finds more of the opponent's likely mistakes. Lc0's usual rule (PUCT) gives most visits to the move that currently looks best, and plays the move with most visits. So whichever move takes the lead early gets more visits, looks better for it, and keeps the lead. From the start position, `1.Nf3` took 127,000 of 135,000 visits and rose from 0.10 to 0.63. Its rivals stayed near their first estimates. Given 51,000 nodes each with `searchmoves`, the moves' values were:

| Move | Value (Q) |
|---|---|
| `1.e4` | 0.509 |
| `1.d4` | 0.461 |
| `1.Nf3` | 0.449 |
| `1.c4` | 0.409 |

These were measured again after the stream fix (see "One CUDA stream"), and `1.e4` still came out best. The other numbers in this paragraph come from before it. Raising the root's exploration (`RootHasOwnCpuctParams`, `CPuctAtRoot` 5 or 8) only changed which move won the race: one move still took 93% of the visits.

**Sequential halving.** With the opponent model on, the root uses sequential halving, as in Gumbel MuZero, instead:
- **Rounds.** The `OpponentRootCandidates` moves with the highest priors (8 by default) get equal visits. When all of them reach the round's target, they are ranked by value, the worse half is dropped, and the rest go on to the next round.
- **Runs.** A run spreads a budget of visits evenly over its rounds, `ceil(log2(candidates))` of them. When one move is left, a new run starts with twice the budget, so the search can go on for any time. The new run's targets are absolute visit counts, so moves already above them take no visits until the others catch up. The first run starts with the `OpponentRootCandidates` moves. Each later run starts only with the moves that survived the previous run's first round, but never fewer than 4, so that a sharp move whose value rises slowly with search can still overtake. Starting every run with all 8 had spent half of a long search on hopeless moves. After 2M nodes in the Belgrade Gambit position below, `5.Bd3`, `5.Bf4`, `5.Bd2` and `5.Be3`, scoring 0.28 down to −0.32, had 131k visits each.
- **The best move** is the best-valued candidate of the current round, since the candidates take visits in turn and stay within a few visits of each other. When their visits differ by more than 5%, the move ranked best at the last halving is played instead. That happens just after a new run starts, while the dropped moves catch up, or when the candidates bring visits from the previous move's search. Either way, moves are only compared at equal visits. The reported score and principal variation follow the best move, unless a proven win or loss decides.

- **No tree reuse.** Lc0 normally keeps the part of the previous search's tree that is still relevant. Those visits are uneven: they are concentrated on the moves the previous search expected. A round's targets are absolute visit counts, so a candidate that brings 200,000 visits has already passed the early rounds' targets while its rivals are compared with it at far fewer. Since values rise with visits, it wins those comparisons unfairly. In 30 seconds of infinite analysis after `1.e4 e5 2.Nf3 Nc6`, following 30 seconds after `1.e4 e5` as a GUI would, the reused tree ended with `3.Bc4` and `3.d4` (0.734 and 0.733). `3.Nc3` had been dropped a round earlier at 0.682, with half their visits. A fresh tree ended with `3.Nc3` ahead of `3.d4`, 0.734 against 0.718. So with the opponent model on, every search now starts with a fresh tree, unless `OpponentTreeReuse` is set. The same sequence then also ends with `3.Nc3`.

  The network caches carry over too. Lc0 caches a network's result by the position alone, while both networks also read the 8 positions before it. So a result the previous search cached for the same position, reached by another move order, differs a little from a fresh one. At 100,000 nodes after `2.Nf3 Nc6`, two fresh processes both gave `3.Nc3` 0.645 and `3.d4` 0.629. After an analysis of `1.e4 e5` in the same process, `3.d4` came out at 0.635 and 0.646 in two runs, and it was played once. So both caches are now cleared before each search as well. After an analysis of `1.e4 e5`, `3.Nc3` then scored 0.644 and `3.d4` 0.627, matching the fresh process, and `3.Nc3` was played in all three runs.

- **Scouting.** The candidates were at first the moves with the highest priors, so a move the main network rates low was never searched at all. After `1.e4 e5 2.Nf3 Nc6 3.Nc3 Nf6 4.d4 exd4 5.Nd5 Nxe4 6.Qe2 f5 7.Ng5 Be7 8.h4 O-O 9.Qh5 h6 10.Bc4 Kh8 11.Qg6 Bxg5 12.hxg5 Nxg5 13.Bxg5 Qe8+ 14.Ne7 Qxg6 15.Nxg6+ Kh7 16.Nxf8+ Kh8`, `17.Rxh6+` mates in 2, but with a prior of 1.5% it was not among the 8 candidates and got no visits. Plain Lc0 found the mate in 615 nodes, because its usual rule also tries moves with low priors. Now the root uses that rule for its first `OpponentRootScout` visits, 1000 by default, and the candidates are the moves it visited most. The engine then plays `17.Rxh6+` after 303 nodes, and the opening choices are unchanged.

  A first version always played the last halving's leader. When the search stopped partway through a round, that ignored a later comparison at equal visits. After `1.e4 e5 2.Nf3 Nc6`, a search that reused the previous move's tree played `3.d4`, ranked first when four moves were left. Yet `3.Nc3` led `3.d4` in the final pair, 0.684 against 0.677, at 77,000 visits each. A fresh search played `3.Nc3`. Now both play `3.Nc3`.
- **Stopping.** Lc0's smart pruning stops a search once the most-visited move can no longer be caught. Here, visits no longer decide the move, so the stoppers are given equal visit counts, which turns that check off. A search now runs to its full node or time limit. In timed games that uses more clock time per move.

The state lives in `Search::RootHalving` and is updated whenever the root is picked from, with `nodes_mutex_` held exclusively. The visits go to the candidate with the fewest visits started, up to the next fewest.

**Results**, 300,000 nodes:
- **From the start position,** `1.e4` beats `1.Nc3` in the last round, 0.529 against 0.504, at about 68,000 visits each. `1.d4` and `1.Nf3` drop out in the round before, at 0.466 and 0.454.
- **In the test game,** the engine finds the same moves as with PUCT: `6.e4` after `5...b6` (0.921 against 0.908 for `Nc3`), and `8.Bxa6` after `7...a6` (0.953 against 0.904 for `Nc3`).
- **After `6...Bg4`** in the line where the engine had expected `7.f3 Be7??`, it plays `7.h3` and expects `7...Bh5`. `7.f3` drops out in the second round.

**Sharp moves need more search.** Values rise as a line is searched deeper, because the leaves are scored by the main network as if the opponent will play perfectly from there. The opponent's chances to go wrong only count once the search reaches them. Sharp lines rise faster, because more of their value lies in those chances. So the move chosen depends on how long the engine searches.

After `1.e4 e5 2.Nf3 Nc6 3.Nc3 Nf6 4.d4 exd4`, given equal visits with `searchmoves`, the Belgrade Gambit `5.Nd5` scores 0.617 against 0.655 for `5.Nxd4` at about 9,000 visits each. They are level at about 130,000 each, and `5.Nd5` leads at 429,000, 0.812 against 0.805. Root searches:

| Nodes | Visits per finalist, before / after the later runs shrank | Choice |
|---|---|---|
| 100k | 29k / 37k | `5.Nxd4` |
| 300k | 66k / 101k | `5.Nxd4`, by 0.001 |
| 1M | 230k / 295k | `5.Nd5` |
| 2M | 459k / 599k | `5.Nd5` |

The root cannot compare moves beyond the visits it has.

**Tried and not kept: evaluating the opponent's replies at new nodes.** One explanation was that a new opponent node is valued as if the opponent will defend perfectly, until its less likely replies get visits. So for each new opponent node, the main network also evaluated the opponent's likely replies, up to 6 of them or 90% of the probability. The node was then valued by their evaluations, weighted by the opponent model's probabilities. Values rose by about 0.02 for both moves, but the gap between them changed with search length just as before (−0.033, −0.011 and +0.000 at about 9k, 34k and 129k visits each, against −0.038, −0.011 and +0.000). It cost 22% of the speed, so it was dropped. The cause is not the opponent's next move at the leaves. A remaining suspect is our own nodes: their value is the average of all the moves the search tried there, not the best of them, and the gap between the best move and the rest is larger in sharp positions.

**Tried: deeper search.** Two settings make the search deeper, and neither changed the choice. Each was measured as the gap, `5.Nd5` minus `5.Nxd4`, at about 9k, 34k and 129k visits each; the default gave −0.046, −0.016 and −0.006 in the same run.
- **Less exploration of our own moves** (`CPuct` 0.6 instead of 1.745): −0.024, −0.011 and −0.000. Values rose by about 0.08, and the maximum depth at 300k nodes from 29 to 37.
- **A probability cutoff at opponent nodes** (`OpponentMinProbability` 0.02 or 0.05): −0.049, −0.020 and −0.007 with 2%, and −0.053, −0.018 and −0.005 with 5%, 2 to 4 moves deeper.

Root searches at 100k and 300k still chose `5.Nxd4` with each. Both settings deepen both lines alike. The sharp line's value comes from many separate places where the opponent may go wrong, and finding them takes breadth, which grows with the size of the search.

**Leaning toward our best move works.** In Lc0, a position's value is the average of every result passed up through it. Where the engine is to move, that average includes the visits the search spent trying weaker moves, so it lags behind the best move. It lags most in sharp positions, where the move that punishes a mistake is much better than the rest. Normal Lc0 has the same lag on both sides, and the two partly cancel; here the opponent's side is a deliberate average, so only ours drags the value down.

With `OpponentBestLean` above 0, each position on the path a value takes up the tree recomputes its value from its children:
- **Opponent positions:** the visit-weighted average of their replies, which is the opponent model's expectation, since the visits follow its probabilities.
- **Our positions:** that average, moved by the given fraction toward the value of the most searched move.

The most searched move is used rather than the best-valued one, because a value from a few visits is noisy, and taking the best of noisy values is too optimistic.

The gap `5.Nd5` − `5.Nxd4`, at about 9k, 34k and 129k visits each:

| `OpponentBestLean` | ~9k | ~34k | ~129k |
|---|---|---|---|
| 0.0001 (recomputing only) | −0.047 | −0.021 | −0.010 |
| 0.5 | −0.021 | +0.001 | +0.004 |
| 1.0 | +0.004 | +0.012 | +0.011 |

Recomputing on its own changes nothing; the lean does. At 0.5, `5.Nd5` overtakes at about 34k visits instead of about 130k. At 1.0 it leads at every length, and the gap barely changes between 34k and 129k.

Root searches at 1.0:
- **Choices:** `5.Nd5` at 100k nodes; 0.5 gives a tie there, 0.797 against 0.797. Both settings keep `1.e4` from the start position and `6.e4` and `8.Bxa6` in the test game.
- **Mates:** king and queen against king in 8, 8 and 8 moves, where the fastest were 8, 8 and 7.

The costs:
- **Values run higher:** for example 0.70 from the start position against 0.53, and about 0.99 in won positions, where they leave less room to tell moves apart. In the game-report position, both settings preferred `19...Rd3`, a trap like `19...Rd5`: Stockfish gives it +0.12 after `20.Bf4`, against +6.33 for `19...Re7`.
- **Speed:** the recomputation iterates over each position's children, which costs about 25%.

**Winning sooner: discounting.** Against an opponent who keeps making mistakes, almost every reasonable position is won eventually. So the more a move is searched, the more future mistakes are found, and the value of every decent move creeps toward 1, where it no longer tells moves apart. The goal is to win soon, so with `OpponentDiscount` below 1, a result counts for that factor less for every half-move until the game ends.

The end of the game is placed with the network's moves-left head:
- **Network evaluations:** a position's evaluation v, with M half-moves estimated to be left, counts as v × discount^M.
- **Each step up the tree:** the value is multiplied by the discount once more, so a leaf's value at the root is discounted by its depth plus its moves left. That is an estimate of when the game ends, whatever depth the search stopped at. Discounting by depth in the tree alone would penalize lines for being searched more.
- **Proven results:** a mate in M half-moves counts as ±discount^M instead of ±1.

Moves left are averaged up the tree like values, so they reflect how quickly the opponent model's player is expected to collapse. Values still rise as the search finds quicker wins, but a rise now means an earlier expected end.

The gap `5.Nd5` − `5.Nxd4`, at about 9k, 34k and 129k visits each:

| Settings | ~9k | ~34k | ~129k |
|---|---|---|---|
| No discount, no lean | −0.047 | −0.021 | −0.010 |
| Discount 0.99 | −0.014 | +0.014 | +0.030 |
| Discount 0.99, `OpponentBestLean` 1.0 | +0.040 | +0.063 | +0.065 |
| Discount 0.995, `OpponentBestLean` 1.0 | +0.026 | +0.045 | +0.054 |

After `5.Nd5`, the expected game length falls faster with search than after `5.Nxd4`: to 94 half-moves against 104 at about 129k visits.

Root searches with `OpponentBestLean` 1.0:
- **Choices:** `5.Nd5` at 100k nodes with discount 0.99 and 0.995, and the test game's `6.e4` and `8.Bxa6`. From the start position, `1.d4` with 0.99, ahead of `1.Nc3` by 0.008, and `1.e4` with 0.995.
- **The game-report position:** the trap `19...Rd3`, which the engine expects to end the game sooner than the safe `19...Re7`, because most 1300 players miss `20.Bf4`.
- **Mates:** king and queen against king in 6, 8 and 9 moves with 0.99.

The discounted values are on a different scale: 0.3 to 0.6 is already a large advantage.

**A tactically sharper opponent.** Maia-3 finds clear tactical shots about as often as real 1200-1399 players (see "Tactics" in [MEASUREMENTS.md](MEASUREMENTS.md)), but the engine still leaned on the opponent missing them. With `OpponentEngineMix` above 0, that share of the main network's move probabilities, which a strong player's instincts would give, is mixed into Maia's at every opponent node. That costs nothing, since the main network is evaluated at those nodes anyway. It makes the modeled player stronger than measured, by choice.

Root searches, 300k nodes (100k in the Belgrade position):

| Position | Mix 0.25, discount 0.995, lean 1.0 | Mix 0.5, discount 0.995, lean 1.0 | Mix 0.25 alone |
|---|---|---|---|
| Start | `1.e4` 0.319, `1.Nc3` 0.309, `1.d4` 0.264 | `1.e4` 0.312, `1.d4` 0.265 | `1.e4` 0.486, `1.d4` 0.445 |
| Belgrade | `5.Nd5` 0.514, `5.Nxd4` 0.491 | `5.Nd5` 0.465, `5.Nxd4` 0.452 | `5.Nxd4` 0.661, `5.Nd5` 0.647 |
| Game report | `19...Rd3` | `19...Rd5` | `19...Rd5` |
| Test game | `6.e4`, `8.Bxa6` | `6.e4`, `8.Bxa6` | `6.e4`, `8.Bxa6` |

From the start position, `1.e4` now wins with every setting, by 0.04 to 0.06 over `1.d4`. Without the mix, `1.d4` and `1.e4` had been within 0.01. In the game-report position, the traps `19...Rd3` and `19...Rd5` still beat the safe `19...Re7`: a sharper opponent finds `20.Bf4` more often, but the engine still expects to win most of the positions after it.

The main network's instincts are mostly positional, though. On the full-game positions, its probability on the best move beats Maia's by 6.8 points in quiet positions (where the best move leads the next by under 50 cp), but only 1.2 points where the best move leads by 500 cp or more, where it is no better than the players. So the mix mainly makes the opponent defend better positionally.

**A tactical look for the opponent.** Tactics need search. The reweighting from StockfishTrapFinder's `MaiaModel` moves part of each move's probability to a shallow search's best move, more the more the move loses. It acts little in quiet positions, where moves lose little, and strongly where one move is clearly best. The players' and Maia's probability on the best move, and the gain from reweighting at rate 0.7 with a loss scale of 300 cp:

| Best move's lead | Players | Maia | Material only, captures | Material, two plies | Stockfish 11, depth 1 | Stockfish 11, depth 2 |
|---|---|---|---|---|---|---|
| 0-50 cp | 25.7% | 24.6% | +0.6 | +0.8 | +1.8 | +1.3 |
| 200-500 cp | 57.8% | 60.2% | +3.9 | +5.0 | +8.0 | +7.5 |
| 500+ cp | 76.1% | 74.9% | +4.3 | +7.8 | +10.9 | +10.2 |

Stockfish 11 is the last Stockfish with a hand-written evaluation (material, piece placement, mobility, king safety, pawns), closer to a club player's judgment than a network's. At depth 1 it gives the largest tactical gain, while quiet positions move about as little as with the others. Depth 2 adds nothing.

So `OpponentTactics` scores every move at each new opponent node with a depth-1 search: the move, then the capture sequences (all moves in check), with captures that lose material by exchange skipped, judged by Stockfish 11's classical evaluation. The opponent model's probabilities are then reweighted with the option as the rate. Stockfish 11's board, move generation and evaluation are built into Lc0 from [lc0-src/third_party/stockfish11/](lc0-src/third_party/stockfish11/README.md), behind `ScoreOpponentMoves` in `opponent_tactics.cc`. Its scores match the Stockfish 11 engine's at depth 1.

Two limits:
- **Misreading opening traps.** After `1.e4 e5 2.Nf3 Nc6 3.Bc4 Nd4 4.Nxe5`, Maia gives `4...Qg5`, which wins material, 96%. Depth 1 sees `5.Nxf7`, attacking the queen and rook, and scores `4...Qg5` at −305 against −120 for `4...Nh6`, so at rate 1.0 `4...Qg5` drops to 52%. Stockfish 11 needs depth 3 to rank it best, which would cost 50 to 100 times as much.
- **Little to add in the opening.** By phase, at rate 0.7, on big shots (200+ cp): in the opening, players 72.6%, Maia 74.4%, reweighted 80.3%; in the middlegame, 70.3%, 69.3% and 77.8%; late in the game, 57.6%, 60.1% and 73.3%. Maia, which has seen the common traps many times, is already as sharp as the players in the opening.

Opening tests, with discount 0.995 and `OpponentBestLean` 1.0, 300k nodes (100k in the Belgrade position):

| Position | `OpponentTactics` 0 | 0.5 | 1.0 |
|---|---|---|---|
| Start | `1.e4` 0.387, `1.d4` 0.354 | `1.e4` 0.358, `1.d4` 0.346 | `1.e4` 0.387, `1.d4` 0.384 |
| Belgrade | `5.Nd5` | `5.Nd5` | `5.Nd5` |
| `1.e4 e5 2.Nf3 Nc6` | `3.Nc3` | `3.d4` | `3.d4` |
| `1.e4 c5` | `2.d4` | `2.d4` | `2.Nf3` |

It costs about 15 to 25% of the speed.

Where it matters: in the Belgrade line `5.Nd5 Nxe4 6.Qe2 f5 7.Ng5 Bb4+ 8.Kd1 O-O 9.Qh5 Nxg5 10.Bxg5 Qe8`, with both ratings 1400, caution 0.15, discount 0.995 and lean 1.0, the engine played `11.Nf6+?` at 300k nodes (+0.704, against +0.689 for `11.Qxe8`). It expects `11...gxf6` (Maia: 73%) and values the position after it at +0.65, where Stockfish has −7.2: after `12.Bc4+` Maia gives `12...Kh8` 75% and `12...Kg7` 10%, both losing, and the refutation `12...d5` only 8%. With `OpponentTactics` 0.5 the depth-1 look sees the king moves lose, `12.Bc4+` drops to +0.21 and `11.Nf6+` to +0.28, so it plays `11.Qxe8`; at 1.0, −0.37 and −0.14. `OpponentEngineMix` barely changed it (`12.Bc4+` +0.59 at 0.25, +0.54 at 0.5), since the main network's instincts miss `12...d5` too. Raising `OpponentCaution` to 0.2 or 0.3 also chose `11.Qxe8`, with `11.Nf6+` at +0.644 and +0.525. The Belgrade's `5.Nd5` and `1.e4` stayed with all of these except caution 0.5 and engine mix 0.5, which played `5.Nxd4`.

**Opening choices, and two things that did not change them.** In games between players rated about 1250-1449 from the games file, White's score depends on the opening: the Scotch (`1.e4 e5 2.Nf3 Nc6 3.d4`) scores 55.5%, the Spanish 53.1%, the Italian 52.1% and the Four Knights (`3.Nc3`) 50.8%; `1.d4` 52.5% and `1.e4` 51.7%. The engine, which with discount 0.995 and `OpponentBestLean` 1.0 plays `3.Nc3`, seemed to be missing the long-term signal in such statistics.

- **The opponent model's own evaluation.** Over 986 positions in the first 8 plies with at least 2,000 games each, Maia-3's value (both players 1300) ranked them by White's real score with a rank correlation of 0.71, against 0.14 for the main network's value. Blending it into the value of new opponent nodes, half or all of it, left the choices in four opening positions unchanged (`3.Nc3` after `2...Nc6`, `2.d4` after `1...e5` and after `1...c5`), lowered every value, and was not kept. A line's value at the root averages thousands of positions deep in the tree, reached by the engine's moves, not a 1300 player's.
- **More search.** With equal search, `3.Nc3` leads `3.d4` at every length: 0.481 against 0.454 at about 34k visits each, 0.537 against 0.513 at 129k, 0.585 against 0.566 at 429k.

The engine's main lines show why: `3.Nc3` is not the quiet Four Knights the statistics mostly count, but the Scotch Four Knights, `3...Nf6 4.d4 exd4 5.Nxd4 Nxd4 6.Qxd4`, where it expects natural mistakes such as `6...b6?`, met by `7.e5`. After `1...e5` it plays `2.d4`, reaching the Scotch by transposition, and after `1...c5` an open Sicilian by `2.d4`. Its opening choices are sharp central lines, chosen consistently, so they reflect its model of the opponent rather than a lack of search.

### Caution: refutations that are plausible
The engine values a position by the expected result against the opponent model, so it will give up material for a line in which the model's player usually goes wrong, however close at hand the refutation. In a Halloween Gambit game (`1.e4 e5 2.Nf3 Nc6 3.Nc3 Nf6 4.Nxe5 Nxe5 5.d4 Ng6 6.e5 Ng8 7.Qf3 d5`), with both ratings at 1500, it played `8.Nxd5?` (Stockfish: from −2.3 to −6.1), expecting `8...Be6` (47.5%), when `8...c6`, which the model gave 28%, refutes it. Later it played `15.Rxe7+?` from a position Stockfish scores −6.9, which it valued at +0.54, an expected score of about 77%. Maia-3's own value for that position, between two 1500 players, gives White 25%. Following the engine's main line, its Black moves were mostly Maia's top choices. Several were mistakes, such as walking into a second knight fork at 46%, so the engine was applying its model faithfully. But a real player who is a piece up, facing precise threats, may be more careful than the model's move-by-move rates. The lean toward our best move adds to the optimism (+0.40 without it), but is not the cause (+0.54 without the lean or the discount).

The aim is to keep tricks whose refutation is very unlikely, and avoid those a player may well find. So with `OpponentCaution` above 0, the value of a position where the opponent is to move leans by that fraction from the expected value toward the value of the opponent's best plausible reply: the best of those the model gives at least `OpponentRefutationProbability` (5%). It is computed in the same recomputation of values from children as `OpponentBestLean`. A move refuted only by a reply rarer than that keeps its value; a trap whose plausible best reply leaves the position level keeps most of it.

With both ratings at 1500, discount 0.995 and `OpponentBestLean` 1.0, 300k nodes:

| Position | Caution 0 | Caution 0.3 | Caution 0.6 |
|---|---|---|---|
| Halloween, move 8 | `8.Nxd5` 0.416 | `8.Qxd5` 0.188 (`8.Nxd5` 0.107) | `8.Qxd5` |
| Halloween, move 15 | `15.Bg7` 0.546 (`15.Rxe7+` 0.454) | `15.Bg7` 0.267 (`15.Rxe7+` 0.062) | `15.Bg7` (`15.Rxe7+` −0.33) |

`8.Qxd5` is Stockfish's choice; `15.Bg7` and Stockfish's `15.Qxf5` both score −6.9, against −9.1 for `15.Rxe7+`. Values are lower: caution 0.3 takes about 0.15 to 0.2 off them in these positions.

**No cliff at the threshold.** With a hard threshold, a refutation at 4.9% would not count at all, so the engine could sacrifice material expecting a capture the model gives 95%, as if the opponent had to capture, as in checkers. On the full-game positions, Maia-3's confident predictions are well calibrated, captures included: 96.6% predicted against 95.9% played at 95-98%, and 99.2% against 99.0% above 98%. On captures that lose material at depth 10, which are poisoned, it predicts 27.0% against 27.0% taken at 150-299 cp, and 21.5% against 18.6% at 300 cp or more. But a player facing an engine's deliberate sacrifice may be more suspicious than in games between humans, which this data cannot show.

So each reply counts with a weight of its probability over the threshold, at most 1: the value leans by `OpponentCaution` toward the reply whose weighted gain over the expected value is largest. A reply at 4% then counts with 0.8 of the weight at the 5% threshold, and one at 0.1% with 0.02. A lower threshold is more cautious, and a higher one relies more on unlikely refutations being missed.

With caution 0.3 and the threshold at 5%: `8.Qxd5` and `15.Bg7` again, `1.e4`, `2.d4` after `1...c5`, and the mate in 2 after 306 nodes. The Belgrade Gambit gives way to `5.Nxd4` at 100k nodes (0.338 against 0.306), and `3.Bc4` edges `3.Nc3` after `2...Nc6` (0.318 against 0.308). The Belgrade's good replies for Black are not rare enough to ignore.

**But caution 0.3 is too much for the Belgrade.** In the games file, across all ratings, White scores 60.7% after `5.Nd5` (546 games) against 53.8% after `5.Nxd4` (17,235): 55.9% against 52.9% where the average rating is under 1600, and 67.6% against 55.6% from 1600. Players who choose the Belgrade likely know it, which flatters it, but the edge is clear. After `5.Nd5`, Black played `5...Nxd5` in 30% of the games (White scored 61%), `5...Nxe4` in 27% (68%), `5...Bc5` in 18% (56%), and `5...Bb4`, the best defense, in 7% (49%). The good replies equalize, they do not refute. Caution penalizes any move whose plausible best reply falls well below its expected value, so it held down a gambit with a large upside and an equal worst case. Weaker caution fixes the Halloween moves and keeps the Belgrade:

| Caution, threshold | Belgrade (100k nodes) | Halloween, move 8 | Halloween, move 15 |
|---|---|---|---|
| 0.1, 5% | `5.Nd5` | `8.Qxd5` | `15.Bg7` |
| 0.15, 5% | `5.Nd5` | `8.Qxd5` | `15.Bg7` |
| 0.2, 5% | `5.Nd5` | `8.Qxd5` | `15.Bg7` |
| 0.3, 15% | `5.Nd5` | `8.Qxd5` | `15.Bg7` |
| 0.3, 30% | `5.Nd5` | `8.Qxd5` | `15.Qxf5` |

With caution 0.15 the start position (`1.e4`), `3.Nc3` after `2...Nc6`, `2.d4` after `1...c5` and the mate in 2 are unchanged. So 0.15 is the suggested setting.

**More caution when ahead (`OpponentCautionAhead`).** A result is at most a win, so once the engine is well ahead against the model, a line that gives the advantage back for a likely mistake scores about the same as one that keeps it, and noise decides. In the Belgrade line `5.Nd5 Nxe4 6.Qe2 f5 7.Ng5 d6 8.Nxe4 fxe4 9.Qxe4+ Be7 10.Bg5 h6 11.Bh4 g5`, with ratings 1400, caution 0.15, discount 0.995, lean 1.0 and `OpponentTactics` 0.5, it played `12.Be2` (Stockfish 0.00) over `12.Bg3` (+3.6) at 300k nodes, +0.852 against +0.845. Without the opponent model the two were +0.04 and +0.78. The risk is lopsided too: the sacrifice could gain at most about 0.15, and lose about 0.8 wherever the model is wrong about the position. So at each position where the opponent is to move, the caution is raised by `OpponentCautionAhead` times this engine's expected result there, when that is positive. In this line, at 0.25, 0.5 and 1.0 the engine played `12.Bg3` by 0.013, `12.Bg3` by 0.115, and `12.Bb5`, and at 1.0 the Belgrade gave way to `5.Nxd4`.

[tools/test_caution_ahead.py](tools/test_caution_ahead.py) took 200 positions from real games, in every phase, where the side to move is ahead by 1.5 to 10 pawns (Stockfish, depth 12), and scored the engine's moves at 50k nodes with the settings above (Stockfish, depth 16):

| `OpponentCautionAhead` | Drop of 1+ pawns | Drop of 2+ | Ended below +1.5 | Lost over half the advantage, ending below +3 | A different move than at 0 |
|---|---|---|---|---|---|
| 0 | 21 | 7 | 15 | 11 | — |
| 0.25 | 16 | 3 | 10 | 8 | 18.5% |
| 0.5 | 10 | 4 | 6 | 5 | 29% |

Most drops of 2+ pawns at 0.5 were in positions won either way (+12.9 to +9.1, mate to +9.3, +9.4 to +7.5). Those at 0 included giving back most of the advantage early in the game: +3.3 to +0.2 at ply 8, +5.8 to +2.3 at ply 12, +1.8 to +0.2 at ply 18, which 0.25 or 0.5 kept. The counts are small, and the run-to-run noise was not measured, but they fall steadily with the option.

The cost is the Belgrade's margin: with equal search, `5.Nd5` − `5.Nxd4` was +0.025 and +0.033 at about 34k and 129k visits each at 0, +0.009 and +0.021 at 0.25, and +0.002 and +0.006 at 0.5. All three still played `5.Nd5` at 100k nodes, `1.e4`, `2.d4` after `1...e5`, `2.Nf3` after `1...c5`, `11.Qxe8` in the line above, `8.Qxd5` and `15.Bg7` at Halloween, and the mate in 2. After `2...Nc6` it played `3.d4` at 0 and 0.25, and `3.Bc4` at 0.5.

### Tried: a horizon for the opponent model
The engine values a line by how it would score, playing on itself, against the model's player, however long that takes. Lc0 expects to outplay a 1400 player from almost anywhere: in the Two Knights line `1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 4.Ng5 d5 5.exd5 Na5 6.Be2 Nxd5 7.d4 Be7`, it played `8.dxe5?` (Stockfish −3.0, against −0.5 for `8.Ne4`) at 300k nodes, and valued the position a rook down after `8...Bxg5 9.Bxg5 Qxg5 10.Qxd5 Qc1+ 11.Bd1 Qxb2 12.Qxa5 Qxa1` at only −0.18. The aim, though, is openings that reach an advantage early. So with `OpponentHorizon` the opponent model chooses the opponent's moves only within that many half-moves of the current position; beyond them the search is Lc0's own, with values averaged and no lean or caution. At 1 and 2 the engine plays nearly objectively (`12.Bg3` over `12.Be2`, `8.Ne4` over `8.dxe5`).

But it did not help. With the settings of the `OpponentCautionAhead` test and `OpponentCautionAhead` 0.5, on the same 200 positions where the side to move is ahead, it ended below +1.5 in 4 at no horizon, and in 6, 6 and 5 at horizons 6, 10 and 16. The same settings without a horizon had given 6 in the earlier run, so the run-to-run noise is about two positions. Shorter horizons hurt the openings: at 6 the engine played `1.d4`, `5.Nxd4` in the Belgrade and `15.Rxe7+` at Halloween, and the Belgrade's `5.Nd5` − `5.Nxd4` gap with equal search was −0.022 and −0.013 at 6, −0.009 and −0.020 at 10, against +0.003 and +0.006 with no horizon. `8.dxe5` was avoided only at 6. The gambles found so far rest on the opponent going wrong within a few moves (Maia gives `10...Qc1+` only 17%), which no horizon removes. The option stays, off by default.

**Retested with the deep look.** The exported tree showed why a horizon is wanted: at the ends of lines the engine's expected score against the model (89% at Stockfish +0.0, 100% at +1.3) runs far above Lc0's own evaluation (53%, 71%), because it counts on the model's mistakes long after the opening. A 300k-node tree from the Belgrade has half its visits 15 half-moves deep or more, 10% at 21 or more. With ratings 1500, caution 0.15, `OpponentCautionAhead` 0.25, `OpponentTactics` 0.5, `OpponentDeepTactics` 0.7 (depth 2, 100 visits), lean 1.0 and discount 0.995 (one run each):

| `OpponentHorizon` | Belgrade gap (34k, 129k) | Belgrade | Two Knights, move 8 | `12.Bg3`, `1.e4`, `3.d4` |
|---|---|---|---|---|
| 0 | +0.026, +0.030 | `5.Nd5` | `8.dxe5` (+0.38, `8.Ne4` +0.26) | all |
| 12 | +0.003, +0.000 | `5.Nd5` | `8.c4` (`8.dxe5` +0.19, `8.Ne4` +0.23) | all |
| 16 | +0.015, +0.017 | `5.Nd5` | `8.dxe5` (+0.32, +0.26) | all |
| 20 | +0.026, +0.027 | `5.Nd5` | `8.dxe5` (+0.36, +0.27) | all |

At 12 the engine gave up `8.dxe5`, whose hopes lie after the twelfth half-move, but the Belgrade's lead shrank to nothing.

### Tried: judging refutations by their sound value
In the Two Knights line above, the refutation of `8.dxe5` is a chain: with Maia at 1900, `8...Bxg5` 80%, then after `10.Qxd5` the refuting `10...Qc1+` 28% (`10...Nc6` 65%), `11...Qxb2` 65% and `12...Qxa1` 95%, after which Black's queen on a1 can still be trapped (`13.Nc3`, where Maia gives the clean `13...Qb2` 10%). The engine valued the position after `10...Qc1+` at +0.04 for White and the one a rook down at +0.16, against −2.9 and −4.5 by Stockfish, so caution saw little to fear in the refutation. Raising the opponent's rating to 1900 did not change the choice. In a game from that line, the user, as Black, kept the advantage from move 12 to about move 29, and lost only after several later mistakes: the engine's hopes were for slow wins.

So with `OpponentSoundCaution`, each node also keeps a sound value (in padding, so nodes stay 64 bytes): at opponent nodes the best plausible reply in full, at this engine's nodes its most searched move, with no later mistakes counted. Caution then judges a plausible reply by its sound value instead of its expected value. With ratings 1500, caution 0.15, discount 0.995, lean 1.0, `OpponentTactics` 0.5 and 300k nodes (the Belgrade gap with equal search at about 34k and 129k visits each):

| Setting | Move 8 | `8.dxe5` | `8.Ne4` | Belgrade gap | Belgrade | Start | After `2...Nc6` |
|---|---|---|---|---|---|---|---|
| Off, `OpponentCautionAhead` 0.5 | `dxe5` | +0.34 | +0.29 | −0.011, −0.010 | `5.Nxd4` | `1.e4` | `3.Nc3` |
| Sound, `OpponentCautionAhead` 0.5 | `Nf3` | −0.08 | −0.01 | −0.040, −0.118 | `5.Nxd4` | `1.Nf3` | `3.Bc4` |
| Sound, `OpponentCautionAhead` 0 | `h4` | +0.06 | +0.13 | −0.034, −0.111 | `5.Nxd4` | `1.Nf3` | `3.Bc4` |
| Sound, caution 0.08 | `Ne4` | +0.24 | +0.27 | −0.014, −0.089 | `5.Nxd4` | `1.d4` | `3.Nc3` |
| Sound, refutation probability 0.25 | `h4` | +0.12 | +0.04 | −0.011, −0.023 | `5.Nxd4` | `1.e4` | `3.d4` |
| Sound, refutation probability 0.35 | `h4` | +0.19 | +0.28 | −0.008, −0.027 | `5.Nxd4` | `1.d4` | `3.d4` |
| Sound, refutation probability 0.5 | `h4` | +0.32 | +0.39 | −0.013, +0.009 | `5.Nxd4` | `1.e4` | `3.d4` |

(The last three with `OpponentCautionAhead` 0.) A hard exclusion of moves Stockfish scores as losing was considered and rejected: a bad move should stay possible when its refutation is unlikely. A higher refutation probability says that with the sound value: the refutation's full objective damage, counted in proportion to its likelihood up to that probability.

**But it penalizes sacrifices.** In a game with sound caution and refutation probability 0.5, `5.Nd5 Nxe4 6.Qe2 f5 7.Nd2 Be7 8.Nxe4 fxe4 9.Qxe4 O-O 10.Bc4 Kh8 11.Bd3 g6 12.h4 d6 13.h5 Bf5`, White was 1.5 to 2.5 pawns up by Stockfish from move 8, and threw it away with `14.Qe2` (−0.2), where `14.hxg6!` (+2.5) gives the queen for a mating attack. Without sound caution the engine played `14.hxg6` (+0.707, against +0.658 for `14.Qe2`); with it, `14.hxg6` fell to +0.496 and `14.Qf4` (Stockfish −6.5) stayed at +0.607. The sound value takes, at each opponent node, the best plausible reply as Lc0's search values it, and a reply in a sacrificial line is valued by a short search that does not yet see the attack, so the best-looking defense is too good for the defender. A minimum over noisy values is biased toward the defender, most where the search is thin, which is where sacrifices and gambits are. Sound caution is not recommended.

It removes `8.dxe5`, but it charges the Belgrade for its objective refutations too, and the engine turns to quieter openings. Lc0's own judgment barely separates the two: without the opponent model it values `5.Nd5` at −0.07 against +0.06 for `5.Nxd4`, and `8.dxe5` at −0.41 against −0.17 for `8.Ne4`, while Stockfish puts the costs at 0.24 and 2.8 pawns. The option stays, off by default.

### Tried: counting to an easily won position instead of the game's end
With `OpponentDiscount`, a result is discounted by the plies to the predicted end of the game, so a sure win with about 70 plies left is worth 0.995^70 ≈ 0.70, an expected score of 85%. In a tree exported from an opening position (with `TreeExportFile`), the line ends Stockfish scored from +1.0 to +7.5 were all at 81-91%, each near its own ceiling, set by the plies left rather than the size of the advantage, and the opponent model had lifted the smaller advantages toward it (+1.04: 84%, where Lc0 alone gives 69%).

The aim is a quick advantage, so the end that matters is a position a player of the modeled strength wins easily from. With `OpponentWinValue`, the main network's value is scaled by 1 / that value, up to a win or a loss, and the discount counts only the plies from the current position. With ratings 1500, caution 0.15, `OpponentCautionAhead` 0.25, `OpponentTactics` 0.5, lean 1.0 and discount 0.995, the Belgrade's `5.Nd5` − `5.Nxd4` gap with equal search (about 34k and 129k visits each) was −0.006 and +0.012 without it, −0.107 and −0.111 at 0.6, −0.053 and −0.062 at 0.8 and −0.079 and −0.043 at 0.9, and it played `5.Nxd4` in all three. `8.dxe5` in the Two Knights line stayed. At 0.6 many `5.Nxd4` lines also ended in won positions (98%) after common mistakes, but the cap is not the cause, as 0.9 shows: the Belgrade's edge came from the moves-left discount, which favors lines whose games end soon, as its mating attacks do.

It does not reach winning positions sooner either. [tools/play_vs_maia.py](tools/play_vs_maia.py) played the engine (30k nodes per move, the settings above) from the start position against Maia-3 at 1500, each of Maia's moves drawn from its probabilities with the same random draws for each setting, 60 games per setting, half with each color, until Stockfish (depth 14) had the engine at +3 or −3, or 40 half-moves. Nearly every game reached +3 (58, 59 and 59 of 60), so how quickly is what differed:

| `OpponentWinValue` | +3 by half-move 10 | by 15 | by 20 | Mean half-move (41 if never) | Faster / slower than off, same game |
|---|---|---|---|---|---|
| 1 (off) | 10 | 36 | 52 | 16.0 | — |
| 0.8 | 6 | 29 | 44 | 17.1 | 17 / 28 |
| 0.6 | 4 | 25 | 40 | 18.3 | 14 / 31 |

With 0.6 the engine took 2.3 half-moves longer on average (95% bootstrap interval +0.2 to +4.4), with 0.8 1.1 longer (−0.8 to +2.9). The opponent here is the engine's own model, which likely blunders more readily against it than real players would. The option stays, off by default.

### A closer look where the search relies on the opponent
The engine steers toward the positions where the model's player most readily goes wrong, so the model's errors matter most exactly there, even if the model is right on average. `OpponentTactics`' one-move look misjudged the Two Knights line both ways: at 1.0 it took `8...Bxg5`, the right start, from 90% to 42% (it sees `8...Bxg5 9.Bxg5 Qxg5 10.Qxd5` as even), raised `10...Qc1+` from 19% to 43%, and lowered `8.Ne4` more than `8.dxe5` (+0.008 against +0.392).

So with `OpponentDeepTactics`, once a position where the opponent is to move has `OpponentDeepTacticsVisits` visits (1000), CPU threads beside the search score each of the opponent's moves there with an alpha-beta search `OpponentDeepTacticsDepth` half-moves deep (checks extend it, then the capture search), with Stockfish 11's classical evaluation, and the probabilities are reweighted as with `OpponentTactics`. Positions are found again by their path from the root. At depth 3 a position took 38 ms, at depth 4 about 300 ms and at depth 5 about 2.5 s; at depth 4 with 8 threads all of the 180-200 positions a 300k-node search queued were done, and the search's speed did not change.

With ratings 1500, caution 0.15, `OpponentCautionAhead` 0.25, `OpponentTactics` 0.5, lean 1.0 and discount 0.995, at 300k nodes:
- **Belgrade:** `5.Nd5` − `5.Nxd4` with equal search (about 34k and 129k visits each) went from −0.004 and +0.010 to +0.020 and +0.026 at 0.7, depth 4.
- **Unchanged:** `1.e4`, `3.d4` after `2...Nc6`, `11.Qxe8`, `12.Bg3`, the Halloween's `8.Qxd5` and the mate in 2.
- **`8.dxe5` stayed** at every setting tried (0.7 and 1.0, depth 3 to 5, 500 or 1000 visits): +0.34 to +0.37 against +0.26 to +0.30 for `8.Ne4`. The look raised `10...Qc1+` from 31% to 55%, but after `10...Qc1+ 11.Bd1 Qxb2 12.Qxa5 Qxa1 13.Nc3` the engine still expects `13...b6` (39%) or `13...O-O` (42%) to go wrong in lines with fewer visits, and the alternatives' values fell as well.

**With real Stockfish instead.** The closer look first used its own alpha-beta search on Stockfish 11's classical evaluation, which proved far weaker than its depth suggests. After `1.e4 e5 2.Nf3 Nc6 3.Nc3 Nf6 4.Nxe5 Nxe5 5.d4 Ng6 6.e5 Ng8 7.Bc4 c6 8.Qf3 Qe7 9.O-O b5 10.Bb3 Nh4 11.Qd3`, it ranked `11...Nxg2` best at depth 2 (−17 cp for Black, against −200 for `11...d5`) and `11...f5` at depth 4, where Stockfish 19 has `11...d5` (−89) well ahead of `11...d6` (−191) and `11...Nxg2` (−393); at depth 2 and 100 visits it raised `11...Nxg2` from under 6% to 24%, then `12.Ne4 Ne1` (Stockfish +8.95) to 44%. So the closer look now runs Stockfish itself (`OpponentStockfishPath`), one process per thread, scoring every move with MultiPV at `OpponentDeepTacticsDepth` (8 by default: Stockfish 19 needs 80-150 ms a position on one thread, and finds `11...d5` from depth 6). There, `11...d5` became 49% and `11...Nxg2` dropped out. Depth 8 turned out to be more than needed: Stockfish 19 (one thread, MultiPV for all moves) ranks `11...d5` first at every depth from 1 to 8 (9 to 163 ms), and after `11...Nxg2 12.Ne4` never ranks `12...Ne1` near the top. At depths 1-3 the scores swing (`11...d5` +242 at depth 1, −117 at 4), which would move probability about at random, since the reweighting uses each move's loss to the best; from depth 4 (about 15 ms) they settle. So the default is depth 4.

**First game with it.** With ratings 1500, lean 1.0, `OpponentMinProbability` 0.002, discount 0.995, `OpponentTactics` 0.25, caution 0.15 (refutation probability 0.05), no `OpponentCautionAhead`, `OpponentHorizon` 16, and the Stockfish look at 0.5 (depth 6, 500 visits, 8 processes), 300k nodes, the engine as White played a gambit of its own for the first time and won in 14 moves: `1.e4 e5 2.d4 exd4 3.Nf3 Bc5 4.c3 dxc3 5.Bc4 cxb2 6.Bxb2 Nf6 7.e5 Qe7 8.O-O O-O 9.exf6 gxf6 10.Nc3 d6 11.Nd5 Qd8 12.Bxf6 Qd7 13.Qc1 Qg4 14.Ne7#`, two pawns for development as in the Danish Gambit. In further games with these settings the engine consistently found better traps and openings, its suggested lines showing the strongest attacking moves and the Danish-like pawn sacrifice several moves from the root.

With ratings 1500, caution 0.15, `OpponentTactics` 0.25, `OpponentMinProbability` 0.002, lean 1.0, discount 0.995 and the Stockfish look at 0.8 (depth 8, 100 visits, 8 processes), at 300k nodes: with no horizon, the Belgrade gap was +0.010 and +0.021 (`5.Nd5`) but `8.dxe5` stayed (+0.38 against +0.29 for `8.Ne4`); with a horizon of 16, `8.dxe5` fell to +0.19 (`8.h4` played, `8.Ne4` +0.29) and the Belgrade was a tie (−0.007, +0.005; `5.Nxd4` at 100k nodes); with a horizon of 10, `5.Nxd4` (−0.024, −0.034).

**A gamble when far ahead, again.** In `1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 4.d4 exd4 5.O-O Nxe4 6.Re1 d5 7.Bxd5 Qxd5 8.Nc3 Qc4 9.Nd2 Qe6 10.Nb5 Qd7 11.Nxe4 Be7 12.Bf4 O-O 13.Nxc7 Rb8 14.Qf3 Kh8 15.Qg3 Bb4 16.c3 dxc3 17.bxc3 Ba5 18.Ne8 Qxe8 19.Bxb8 f5 20.Ng5 Qg6`, with ratings 1700, `OpponentModelTemperature` 1.5, caution 0.1, no `OpponentCautionAhead`, discount 0.998, `OpponentDeepTactics` 0.5 (depth 4, 100 visits) and 100k nodes, the engine played `21.Qd6`, after which only `21...Qf6` holds (Stockfish: from +4.9 to −0.1); `21.Bd6` and `21.Bf4` keep +4.8. The closer look did its part: Maia gives `21...Qf6` under 2%, the look made it 27%, and Stockfish's other replies there are mates in one (scored −10000, correctly). But the engine still expected White to win after it (lines ending at 100% where Lc0's own evaluation is 40-67%), and every winning move scored about the same (`21.Qd6` +0.881-0.889, `21.Bf4` +0.874-0.875). With `OpponentCautionAhead` 0.25 it played `21.Bf4` or `21.Be5`, and with temperature 1.0 as well `21.Bd6` or `21.Bf4` (`21.Qd6` +0.771); temperature 1.0 alone did not change it. A stronger closer look also removed it: at 0.8 the engine played `21.Be5` twice (`21.Qd6` +0.818 to +0.833, `21.Bf4` +0.867 to +0.880), at 1.0 `21.Bd6` and `21.Be5`; at 0.5 a reply keeps at least half its probability however badly it loses, so Black still misses `21...Qf6` about 73% of the time in the model. The exported tree had not shown `21...Qf6`: it listed replies by visits, and a reply the look raises late has few; it now lists them by probability, and shows a reply of 5% or more even with few visits.

### Tried and not kept: Gumbel selection below the root
Below the root, this engine's moves are chosen by PUCT, which favors whichever move took the lead first. So the lines the search assumes may not be those the root would choose, and the line it shows often seemed to change once its moves came to be played. Gumbel MuZero (Danihelka et al., 2022), whose root rule is the sequential halving used here, chooses below the root by an improved policy instead: pi'(a) = softmax(log P(a) + sigma(q(a))), with sigma(q) = (50 + max N) × c_scale × q, q mapped to [0, 1], and unvisited moves valued at the prior-weighted mean of the visited ones. Visits go to the move furthest below its share of pi', as at opponent nodes.

With both ratings at 1500, discount 0.995, `OpponentBestLean` 1.0 and `OpponentCaution` 0.15, the gap `5.Nd5` − `5.Nxd4` at about 9k, 34k and 129k visits each:

| Selection below the root | ~9k | ~34k | ~129k | Nodes per second |
|---|---|---|---|---|
| PUCT | −0.005 | +0.012 | +0.030 | 21,000 |
| Gumbel, c_scale 0.02 | −0.051 | −0.013 | +0.004 | 16,600 |
| Gumbel, c_scale 0.1 | +0.000 | +0.004 | +0.011 | 16,000 |
| Gumbel, c_scale 1.0 | −0.014 | −0.004 | +0.010 | 16,400 |

The line's own third move was played once its first two had been in 4 of 5 opening positions with PUCT, and with Gumbel at 0.1 too, at 100k nodes. Gumbel at 0.1 also moved the start position to `1.d4` and the Belgrade position to `5.Nxd4`. It did no better and cost about 22% of the speed, so it was not kept.

### Fine-tuning the main network: feasibility
The main network judges positions as if both sides play perfectly, which is why the engine sees a sharp line's practical value only where its search reaches the opponent's mistakes. A network fine-tuned to predict the result against the modeled opponent would include those mistakes in every evaluation. The network, `791556`, has 15 residual blocks of 192 filters with squeeze-and-excitation, an attention policy head, a WDL value head and a moves-left head. Its value head is a 1×1 convolution to 32 channels, then fully connected layers 2048→128→3, about 270k parameters.

Checked so far:
- **ONNX conversion** (`lc0 leela2onnx`, and `onnx2leela` back) is exact: move probabilities within 1 percentage point (half against full precision) and the same values (0.088 against 0.090 from the start position).
- **Lc0's ONNX backend** runs the converted network 5-7 times slower than the native CUDA backend: 13,700 positions per second in fp32 and 18,600 in fp16, against 93,600 at batch 256. Too slow to play with.
- **Lc0's weights format** (`libs/lczero-common/proto/net.proto`) stores each layer as 16-bit values between a minimum and a maximum. With Python bindings generated by `grpcio-tools`, decoding and re-encoding the value head gives identical values on the native backend. Shifting the WDL output's biases moved the start position's value from 0.088 to 0.338 as expected.

So a fine-tuned value head, trained in PyTorch with the body frozen, can be written back into the native weights file and played at full speed. Training data would come from this engine's games against a Maia opponent of somewhat higher rating than the one modeled, so that the network does not learn to rely on one model's particular mistakes.

**Pilot: learnable, but no better decisions.** [tools/finetune_value_head.py](tools/finetune_value_head.py) took 3,000 positions from real games, all phases, and used as targets the opponent-model search's values: 5,000 nodes, opponent and own rating 1700, caution 0.15, no lean or discount. It trained the value head's two fully connected layers on 80% of the positions, with the body and the value convolution frozen, and wrote them into a copy of the network.

On the other 20%, predicting the search's value (W − L):

| Head | Correlation | Mean absolute error |
|---|---|---|
| Original | 0.936 | 0.172 |
| Original, rescaled to fit | 0.936 | 0.133 |
| Fine-tuned | 0.965 | 0.092 |

So the values are learnable beyond an offset. But with the test settings (ratings 1500, caution 0.15, discount 0.995, lean 1.0), the fine-tuned network decided no better:
- **Belgrade, equal search:** the gap `5.Nd5` − `5.Nxd4` was +0.006, +0.010 and +0.017 at about 9k, 34k and 129k visits each, against −0.006, +0.013 and +0.030 for the original. At 100k nodes it played `5.Nxd4`.
- **Halloween, move 8:** it played the unsound `8.Nxd5` again.
- **Other choices:** `1.d4` from the start and `2.d4` after `1...e5`.

Trained on the search's own values, the network takes on the search's optimism in every evaluation, and the search compounds it. The original network's choices also varied between runs in close positions (`2.Nf3` after `1...e5`, `8.exd6` at Halloween move 8), but that does not explain the Belgrade and `8.Nxd5` together. Training on game results against a stronger Maia would avoid using the search's values, but would need thousands of games for a noisy signal.

**Cautious value targets: safe, but nothing to gain.** With targets from a more cautious search (ratings 1900, caution 0.5), the original head already predicted the search's values well (correlation 0.953, mean absolute error 0.101), and fine-tuning improved that only to 0.970 and 0.085. With the test settings the fine-tuned network kept `5.Nd5` and `8.Qxd5`, and its Belgrade gap was about the original's (+0.023 against +0.031 at 129k visits). A search that cautious says little the network does not already know.

**Policy head: learnable a little, and worse decisions.** [tools/finetune_policy_head.py](tools/finetune_policy_head.py) took the same 3,000 positions with targets from the test settings themselves (ratings 1500, caution 0.15, discount 0.995, lean 1.0) and trained the policy head's projection, queries and keys toward half the original policy and half the search's choice (the best-valued of its most visited moves). Training on the root's visits instead was tried first and taught a flatter policy, since sequential halving gives the finalists equal visits. On the 600 held-out positions the probability on the search's choice rose from 0.367 to 0.403 and its share as the top move from 55% to 57%, with the best held-out loss after 4 epochs. Priors moved toward `1.e4` (22.1% to 34.9%) and a little toward `5.Nd5` (3.2% to 5.0%), but also toward `5.Nxd4` (67.5% to 71.2%). With the test settings:
- **Belgrade, equal search:** the gap `5.Nd5` − `5.Nxd4` was −0.043, −0.014 and +0.013, against −0.003, +0.013 and +0.031 for the original, and at 100k nodes it played `5.Nxd4`.
- **Other choices** were the original's: `1.e4`, `2.Nf3`, `3.Nc3`, `2.d4` against the Sicilian, `8.Qxd5` and `15.Bxg7` at Halloween, and the mate in 2.

The policy is used at every one of this engine's moves in the tree, so a head nudged toward 2,400 root choices also changes how the search plays out each line below the root, and the Belgrade's lines got worse. The search's choice depends on the opponent's likely mistakes several moves ahead, which one position's features predict only slightly, as the small held-out gain shows. Neither head was improved by training on this search's own output, and the fine-tuning was not pursued further.

## Open Questions
- **Which main network.** The Lc0 release ships with a network (`791556.pb.gz`) that is good enough to develop with. A stronger network, such as a BT4 transformer network, can come later.
- **Calibration.** A previous project reweighted Maia's probabilities with a low-depth Stockfish evaluation, moving part of a move's probability to the evaluation's best move by `max_rate × (1 − exp(−loss / scale_cp))`, to suppress blunders a quick look would catch. Whether that predicts real moves better can be measured offline with the measurement tool before building it in. Building it in would need a low-depth evaluation of every candidate at each new opponent node, either a Stockfish evaluator compiled into Lc0 (CPU) or the main network's evaluation of the candidate positions (GPU). **Resolved: not needed.** Measured on 7,451 positions from full blitz and rapid games between players rated 1200–1399, raw Maia3-23M at temperature 1.0 already matches how often players blunder, in the opening, the middlegame and late in the game. No calibration setting predicted moves better (see "Tuning the calibration" in [MEASUREMENTS.md](MEASUREMENTS.md)).
- **Ratings outside Maia's range.** Maia covers 1100 to 1900. Maia-3, now supported, covers 600 to 3000.
- **Which Maia-3 size.** The 23M network predicts practically as well as the 79M one at 2.4 times the speed. The 5M one is another 1.6 times faster and predicts a little worse (51.4% top 1 against 52.3%).
