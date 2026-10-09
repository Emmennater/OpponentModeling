# Opponent Modeling Using Stockfish
The problem with modern chess engines is that they assume the opponent will always play the best move. The goal of this project is to modify the search algorithm used by stockfish to handicap the opponent's moves. The problem is that changing the algorithm significantly might disrupt existing optimizations, leaving us to design our own engine. To get around this, the idea is to restrict the move list to a set of moves the opponent would reasonably play.

## Opponent Move Restriction
To decide which moves shall be kept for the opponent, we use a low-depth search. This search will not be used to find moves our opponent will play, but rather moves the opponent will NOT play. 

For example, if a depth 4 search sees move X as a blunder, our opponent will also likely assume X is a blunder. However, if X turns out to be a good move, and even better, the only good move for our opponent, then we can deduce that our opponent will blunder in this position.

First, our low-depth search will identify the top move in the position. Next, we filter the rest of the moves by comparing their evaluation to the top move. If the move is not within 200 centipawns of the top move, it is discarded.

## Integrating with Stockfish
Since we are only changing the set of legal moves for our opponent, the rest of the optimizations should still work. It is worth noting that the set of moves will never be empty, unless the original set of moves is also empty.

It is worth noting that the performance of stockfish will degrade dramatically as the opponent depth is increased. The usual move generation process only takes a few nanoseconds, but it will take much longer with additional searches. On the plus side, it will reduce the number of positions considered by the opponent. 

### Quiescence Search
For now, quiescence search is left unchanged: the opponent may still play any capture there. Only the main search restricts the opponent's moves.

## Design of the Low-Depth Search
The step-by-step implementation, with references into the Stockfish source, is in [PLAN.md](PLAN.md).

### A Search That Depends Only on the Position
The low-depth search is a small, separate search: a plain alpha-beta search that looks at every move to the opponent depth, followed by a search of captures. It uses Stockfish's move generation and evaluation network, but has no hash table, no history tables and no pruning beyond alpha-beta. It also ignores repetitions and the 50-move rule, which depend on how a position was reached.

As a result, the verdict on a move depends only on the position. That matters because a verdict is cached and reused for the rest of the search. A first version used Stockfish's own search function instead. At low depth, Stockfish's pruning depends on history tables that change with every search, so the same position could get very different verdicts. In one position, depending on what had been searched before, the opponent was allowed either 34 of its 47 moves or only one. The main search then built a winning line on that single forced move. PLAN.md has the details.

The search keeps its own copy of the evaluation network's state, and every move it makes on the shared position is undone, so it can't disturb the main search.

### Filtering the Moves
A move is allowed if it scores at least the opponent's best score minus the threshold. To avoid deciding moves the main search never reaches, each move is decided only when needed:

1. When the main search reaches an opponent position, the low-depth search finds the opponent's best score exactly. The first move gets an open search window; each later move is tested against the best so far with a cheap null-window search, and searched fully only if it might be better.
2. When the main search is about to search one of the opponent's moves, a null-window search tests whether that move scores at least `best - threshold`. Moves whose score was already found exactly in step 1, including the best move, need no test.

Because the search looks at every move, the answer to each test is exact, whatever order moves are tried in. The best move is always allowed, so the opponent always has at least one move when it has a legal one.

### Hooking into the Main Search
Three changes are made to `search()`:
- At nodes where the opponent is to move, the opponent's best score is found just before the moves loop, after the `moves_loop` label so that nodes in check are covered too. Nodes that end earlier (hash table hits, razoring, futility pruning, null-move pruning) never pay for a low-depth search.
- In the moves loop, each move is tested just after Stockfish's own pruning, so moves Stockfish would prune anyway are never tested. Disallowed moves are skipped and not counted, so late-move reductions and move-count pruning see only the allowed moves and the moves Stockfish pruned.
- A move that is the opponent's only allowed move gets no singular extension, unless it is also the only legal move. Stockfish extends a move by up to 3 plies when all other moves are much worse, and a move the model forces always looks that way. Extending such moves made lines where the opponent is forced move after move grow deeper instead of shallower.

Each position's best score and verdicts are kept in a filter cache, so a position that comes up again costs nothing. Since every verdict depends only on the position, the cache only saves time; it can't change a result. The cache is cleared on `ucinewgame` or when an opponent parameter changes.

### Known Limitations
- ProbCut tries captures before the moves loop, so it may use a capture the opponent would not play. This errs in the opponent's favor and is left unchanged for now.
- The low-depth search looks at every move, so it is slow compared with the rest of the engine, and its cost grows steeply with depth. See "First Results" below.

## Stockfish Version and Building
The engine is based on the Stockfish 19 release (tag `sf_19`). Its source is kept in [stockfish/](stockfish/), and our changes are made directly to it.

To build on Windows, install [MSYS2](https://www.msys2.org/) with the `mingw-w64-x86_64-gcc` and `mingw-w64-x86_64-make` packages, then run [build.bat](build.bat) from the repository root:

```
build.bat
```

This builds `stockfish\src\stockfish.exe` for the current computer's CPU. To build for other machines, pass an architecture, for example `build.bat x86-64-avx2`. MSYS2 is expected in `C:\msys64`; set `MSYS2_ROOT` if it is installed elsewhere. The evaluation network is downloaded on the first build and embedded in the executable, so `stockfish.exe` can be copied anywhere and loaded into any chess GUI that supports UCI.

`build.bat` puts MSYS2's tools at the front of `PATH`. The Makefile runs shell scripts, and they fail under a shell whose path contains a space, such as Git's `C:/Program Files/Git`.

To check the build, run `stockfish.exe bench`. The unmodified source gives `Nodes searched : 2497913`.

## Checking the Opponent Model
The non-standard UCI command `opponentmoves` runs the low-depth search for the side to move in the current position, using the current `Opponent Depth` and `Centipawn Threshold`. It prints every legal move with its exact score and whether it is kept. For example, in the console:

```
position fen rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3
opponentmoves
```

Here only `f3h4`, winning Black's queen, is kept. Every other move is more than 200 centipawns worse.

The command also runs the faster procedure the main search uses, and prints an error if any of its verdicts differ from the exact scores.

## First Results
**Speed.** Time for `bench` (50 positions searched to depth 13, one thread) at different opponent depths:

| Opponent Depth | Time | Compared with Stockfish |
|---|---|---|
| 0 (disabled) | 2.5 s | 1× |
| 1 | 8.7 s | 3.5× |
| 2 (default) | 71.9 s | 29× |
| 3 | 232 s | 93× |

These compare the time to reach the same depth. With the model enabled, the engine searches a different, restricted tree, so the comparison is only a rough guide. Almost all of the extra time is spent in the low-depth searches: the main search itself visits fewer nodes than plain Stockfish, because the opponent has fewer moves. A first version that used Stockfish's own search for the opponent was much faster (2.3× at depth 2), but its verdicts were unreliable (see "A Search That Depends Only on the Position").

**Traps.** The engine plays moves that set traps for the modeled opponent, and reports scores for the game against that opponent, not objective ones. The scores are only as good as the opponent model: an opponent who sees more than the model will not fall for these traps. With the first version's unreliable verdicts, the engine often claimed +7 in quiet positions. For example, it played `2.Bb5` after `1.e4 c5`, expecting Black to be forced into `2...Qa5`. With the current version, it plays `2.Nf3` there and reports a normal score.

## Measuring the Model Against Real Games
[tools/measure_model.py](tools/measure_model.py) replays games from a file of Lichess games and compares the model with the moves actually played. It needs Python with `python-chess`. Each line of the games file holds the white rating, black rating, result and the moves in SAN, separated by tabs. For example:

```
python tools/measure_model.py lichess_games.tsv --models stockfish,maia --games 300
python tools/measure_model.py lichess_games.tsv --models maia,maia3 --calibrate --truth-depth 4 --games 300
```

For each position, it records the rank of the move that was played among the model's moves, and whether it was the model's top move (its predicted move). For this engine's model (`--model stockfish`), it also records whether the model kept the played move, and the move's centipawn loss at low depth. From the loss, it reports which share of played moves each threshold would keep.

With `--model maia`, it measures the [Maia](https://github.com/CSSLab/maia-chess) networks instead. These networks are trained on Lichess games to predict the move a human of a given rating would play. Each position uses the network for the mover's rating (Maia-1100 to Maia-1900). This needs Lc0 in `lc0/bin` and the networks in `lc0/nets`; both are downloaded separately and not kept in git. The measurements below used the Lc0 v0.32.1 Windows CUDA 12 release.

First results, on 300 games per rating band, for positions from ply 6 to 15. The games file only has the first 16 plies of each game, so these are opening positions. Opponent Depth 2:

| Mover's rating | Kept at 200 cp | Predicted move played | Kept at 20 cp | Kept at 100 cp | Share of legal moves kept |
|---|---|---|---|---|---|
| Below 1200 | 84.7% | 27.0% | 41.6% | 71.4% | 51% |
| 1400–1599 | 94.8% | 34.7% | 56.4% | 87.7% | 57% |
| 1600–1799 | 96.7% | 35.9% | 59.6% | 91.0% | 60% |
| 2200 and above | 99.3% | 51.2% | 74.1% | 95.9% | 61% |
| All | 95.5% | 37.2% | 59.4% | 89.0% | 58% |

- **The 200 cp threshold keeps about 95% of the moves actually played.** It removes about 42% of the legal moves, and those are almost never played. Players below 1200 play a move more than 200 cp worse about 15% of the time.
- **A threshold of 20 cp would miss about 40% of the moves actually played.** The opponent would often play a move the engine never considered.
- **Players play the predicted move only about 37% of the time**, from 27% below 1200 to 51% at 2200 and above. Most of the time they play some other reasonable move.
- **Opponent Depth makes little difference.** Depths 1, 2 and 3 predict the played move about equally well (37.2%, 37.2% and 38.7% overall), so in the opening, depth 1 is as good a model as depth 2, at a fraction of the cost.

### Comparison with Maia
On the same 20,736 positions, Maia predicts the played move much better than this engine's model, in every rating band:

| Mover's rating | Maia top 1 | This model's top 1 (depth 2) | Maia top 3 | This model's top 3 | Maia top 5 | This model's top 5 |
|---|---|---|---|---|---|---|
| Below 1200 | 39.3% | 27.0% | 68.8% | 50.8% | 80.7% | 63.6% |
| 1400–1599 | 48.0% | 34.7% | 77.8% | 63.7% | 87.8% | 75.8% |
| 1600–1799 | 48.4% | 35.9% | 77.8% | 64.0% | 88.5% | 76.5% |
| 2200 and above | 55.0% | 51.2% | 82.2% | 79.5% | 90.5% | 88.4% |
| All | 48.8% | 37.2% | 78.0% | 65.6% | 88.0% | 77.4% |

The gap is largest for weaker players, which is where an opponent model matters most. Maia also gives each move a probability: on average it gave the played move 35.8%. That is the kind of input an expected-value search needs.

Two caveats. Maia was trained on Lichess games, and some of the games in this file may have been in its training data, which would flatter its results. And these are opening positions only.

### Maia-3
`--models maia3` measures [Maia-3](https://github.com/CSSLab/maia3), a transformer that takes both players' ratings as input and covers ratings from 600 to 3000 in one network. It runs in PyTorch, through the `Maia3Net` wrapper from the StockfishTrapFinder project (`--maia3-path`), so the tool must then run under a Python environment with PyTorch and the `maia3` package. On the same positions, all three sizes predict better than Maia-1:

| Model | Top 1, all | Top 1, 1200–1399 | Top 3, all | Log-likelihood, all |
|---|---|---|---|---|
| Maia-1 | 48.8% | 46.1% | 78.0% | −1.61 |
| Maia3-5M | 51.4% | 47.4% | 80.2% | −1.51 |
| Maia3-23M | 52.3% | 49.0% | 80.6% | −1.47 |
| Maia3-79M | 52.6% | 48.9% | 80.7% | −1.46 |

The 23M network is practically as good as the 79M one.

### Calibrating Maia with a low-depth evaluation
`--calibrate` also measures each Maia model with the reweighting from StockfishTrapFinder's `MaiaModel`: moves below 2% probability are dropped, the evaluation's best move is added, and each move gives up `0.7 × (1 − exp(−loss / 200))` of its probability to the evaluation's best move, where `loss` is how far its depth-2 score is below that move's. The aim is to stop the opponent model from counting on blunders a quick look would catch.

It makes the predictions slightly worse: Maia-3's top move matches the played move 50.9% of the time instead of 52.6%, and its log-likelihood drops from −1.46 to −1.91. Most of the log-likelihood drop comes from the 2% cut, which removes moves that are sometimes played. Without it (`--min-prob 0`) the log-likelihood is −1.50.

The more relevant question is whether the models put the right amount of probability on blunders. `--truth-depth 4` compares the probability each model puts on moves losing at least L centipawns, measured at depth 4, with how often players actually played such a move. For players rated 1200 to 1399:

| Model | L = 100 | L = 200 | L = 300 | L = 500 |
|---|---|---|---|---|
| Actually played | 13.6% | 5.2% | 3.0% | 1.0% |
| Maia-1 | 16.1% | 7.1% | 4.1% | 1.9% |
| Maia-1, calibrated | 9.1% | 3.1% | 1.4% | 0.5% |
| Maia3-79M | 14.8% | 6.3% | 3.7% | 1.6% |
| Maia3-79M, calibrated | 8.6% | 2.8% | 1.3% | 0.4% |

Raw Maia slightly overestimates how often players at this level blunder, and Maia-3 less than Maia-1. With these settings, the calibration overcorrects: it predicts less than half the blunders players actually make. A milder setting may land in between.

### Tuning the calibration
[tools/tune_calibration.py](tools/tune_calibration.py) collects the slow inputs once and saves them:
- Maia3-23M's probabilities;
- this engine's scores of every legal move at each calibration depth;
- for the blunder check, the score of every move from a normal Stockfish search (MultiPV) at depth 10.

It then scores a grid of settings from the saved data in seconds:

```
python tools/tune_calibration.py lichess_games.tsv --collect cal.pkl --games 500 --depths 1,2,3,4
python tools/tune_calibration.py --data cal.pkl --depths 1,2,3,4 --temperatures 0.9,1.0 --scales 200,800,3200 --rates 0.2,0.5,0.7
```

Collecting took about 4 minutes for 2,060 positions of players rated 1200–1399. Most of that was the depth-4 scores and the depth-10 truth scores. The results, losses measured at depth 10:

| Model | Log-likelihood | Top 1 | L = 100 | L = 200 | L = 300 | L = 500 |
|---|---|---|---|---|---|---|
| Actually played | | | 12.9% | 5.6% | 3.2% | 1.7% |
| Maia3-23M | −1.537 | 49.5% | 15.1% | 6.4% | 4.0% | 2.1% |
| Maia3-23M, temperature 0.9 | −1.544 | 49.5% | 13.9% | 5.7% | 3.5% | 1.8% |
| Calibrated: depth 1, scale 200, rate 0.3 | −1.543 | 49.9% | 13.4% | 5.3% | 3.2% | 1.6% |
| Calibrated: depth 2, scale 3200, rate 0.2, temperature 0.9 | −1.544 | 49.8% | 13.8% | 5.6% | 3.4% | 1.7% |
| Calibrated: depth 2, scale 200, rate 0.7, 2% cut (the old default) | −1.994 | 48.0% | 9.4% | 3.1% | 1.6% | 0.7% |

- **Raw Maia-3 overestimates blunders by about 20%** at every loss level. At L = 100 the gap is about three standard errors.
- **A mild calibration corrects it,** with a rate of 0.2 to 0.3, or with a large scale_cp. So does a temperature of 0.9 alone.
- **The old default removes about half the blunders players actually make.** Most of its drop in log-likelihood comes from the 2% cut.
- **The calibration depth hardly matters.** Depth 1 does as well as depth 4.
- **No setting predicts the played move noticeably better than raw Maia-3.** The differences in log-likelihood (±0.01) and top 1 (±0.5%) are within the noise of this sample.

So the calibration step does not need to be built into Lc0. `OpponentModelTemperature 0.9` gives the same correction at no cost. As with the other measurements, these are opening positions only (plies 6–15), since the games file stops at ply 16.

## Opponent Modeling in Lc0
The Stockfish version models the opponent as playing the best move, for it, from a restricted set. A second version, built on [Lc0](https://lczero.org), models the opponent as playing like a [Maia](https://github.com/CSSLab/maia-chess) network instead: at the opponent's moves, the search shares out its visits in proportion to Maia's move probabilities, so each position's value becomes the expected result against that opponent. The design and results are in [LC0_PLAN.md](LC0_PLAN.md).

Its source is Lc0 v0.32.1, in [lc0-src/](lc0-src/). To build it on Windows, install the CUDA toolkit, Visual Studio 2022 or its Build Tools with the C++ workload, and `meson` and `ninja` (`python -m pip install --user meson ninja`), then run [build_lc0.bat](build_lc0.bat). The result is `lc0-src\build\lc0.exe`.

It needs two networks: a main network for its own play, and a Maia network for the opponent. These UCI options control the opponent model:
- **OpponentModelWeights:** the Maia network file, for example `lc0/nets/maia-1300.pb.gz`, or a Maia-3 network exported to ONNX, for example `lc0/nets/maia3-23m-fp16.onnx`. Empty turns opponent modeling off.
- **OpponentModelTemperature** (default 1.0): above 1 makes the opponent less predictable, below 1 more.
- **OpponentModelRating** and **OpponentModelOwnRating** (default 1300 each), for Maia-3 only: the opponent's rating, and the rating of the side the engine plays.

Maia-3 networks run on ONNX Runtime. To build with it, unpack ONNX Runtime 1.31's GPU package for CUDA 13 (`onnxruntime-win-x64-gpu_cuda13-1.31.0.zip`) into `lc0/`, and install cuDNN 9, for example with `pip install nvidia-cudnn-cu13` into `.venv` (or set `CUDNN_BIN` to the folder that holds `cudnn64_9.dll`). [build_lc0.bat](build_lc0.bat) then builds with ONNX Runtime and copies its DLLs and cuDNN's next to `lc0.exe`. [tools/export_maia3.py](tools/export_maia3.py) exports a Maia-3 checkpoint, for example `python tools/export_maia3.py maia3-23m lc0/nets/maia3-23m-fp16.onnx --fp16`. The fp16 23M network is the best balance of accuracy and speed: about 24,000 nodes per second, against 69,000 with Maia-1 (see [LC0_PLAN.md](LC0_PLAN.md)).

The scores it reports are expected results against the modeled opponent, so they are much higher than plain Lc0's: a strong engine is expected to beat a 1300 player from most positions.

## Model Parameters
Aside from the usual Stockfish parameters, we have a few for controlling our opponent's moves:
- Opponent Depth (default 2, at most 8): Search depth for our opponent. 0 disables opponent modeling. The low-depth search looks at every move, so its cost grows steeply with depth.
- Centipawn Threshold (default 200): The centipawn threshold used for discarding opponent moves.
- Opponent Hash (default 16): Size in MB of each thread's filter cache, which remembers the opponent's best score and the verdicts on its moves for each position.
