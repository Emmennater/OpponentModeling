# Opponent Modeling Using Maia
The problem with modern chess engines is that they assume the opponent will always play the best move. The goal of this project is to create an engine that doesn't assume that the opponent will always play the best move. We need to identify moves our opponent are likely to make and exploit them by maximizing an expected score.

The project has two engines:
- **The Lc0 version** models the opponent as playing like a human of a given rating, using the Maia networks. It is the more accurate model.
- **The Stockfish version** restricts the opponent to the moves a low-depth search considers reasonable.

| Document | Contents |
|---|---|
| [LC0_PLAN.md](LC0_PLAN.md) | Design, implementation and results of the Lc0 version |
| [STOCKFISH_DESIGN.md](STOCKFISH_DESIGN.md) | Design and first results of the Stockfish version |
| [PLAN.md](PLAN.md) | Step-by-step implementation of the Stockfish version |
| [MEASUREMENTS.md](MEASUREMENTS.md) | How well each opponent model predicts moves in real games, and the calibration tuning |

## The Lc0 Version
Built on [Lc0](https://lczero.org), this version models the opponent as playing like a [Maia](https://github.com/CSSLab/maia-chess) network. At the opponent's moves, the search shares out its visits in proportion to Maia's move probabilities, so each position's value becomes the expected result against that opponent. The design and results are in [LC0_PLAN.md](LC0_PLAN.md).

### Using It
It needs two networks: a main network for its own play, and a Maia network for the opponent, and for the closer look at the opponent's moves, a Stockfish executable. The defaults are the recommended settings, those the engine found its own gambits and traps with against 1500-rated play (see LC0_PLAN.md); set the two networks, **OpponentStockfishPath** and the ratings. These UCI options control the opponent model:
- **OpponentModelWeights:** the Maia network file. It can be a Maia-3 network exported to ONNX, for example `lc0/nets/maia3-23m-fp16.onnx`, or a Maia-1 network, for example `lc0/nets/maia-1300.pb.gz`. Empty turns opponent modeling off.
- **OpponentModelRating** and **OpponentModelOwnRating** (default 1500 each), for Maia-3 only: the opponent's rating, and the rating of the side the engine plays, in Lichess ratings. A higher rating does not make Maia much less prone to tactical blunders; OpponentTactics and OpponentDeepTactics do that.
- **OpponentRootCandidates** (default 8): the number of root moves the engine compares at equal visits, by sequential halving, before choosing its move. With the opponent model, a move's score rises the more it is searched, so the usual rule would favor whichever move took the lead first (see [LC0_PLAN.md](LC0_PLAN.md)). 0 or 1 turns this off.
- **OpponentRootScout** (default 1000): root visits made with Lc0's usual rule before the equal-visit comparison starts. It picks the candidates, so that moves the main network rates low, such as a mating sacrifice, can be among them.
- **OpponentTreeReuse** (default off): whether a search keeps the previous search's tree and network caches. By default every search starts afresh, so an analysis gives the same move whatever was analysed before. Kept visits are uneven and would tilt the equal-visit comparison toward the moves searched most. Cached evaluations of positions reached by other move orders shift values enough to change close calls.
- **OpponentBestLean** (default 1.0): how far the value of a position where the engine is to move leans from the average of the moves it tried there toward its main move. At 1.0, sharp lines such as the Belgrade Gambit get their due at much shorter searches, at the cost of about 25% of the speed and higher evaluations (see [LC0_PLAN.md](LC0_PLAN.md)).
- **OpponentCaution** (default 0.15) and **OpponentRefutationProbability** (default 0.05): how far the engine's judgment of a position where the opponent is to move leans from the expected result toward the opponent's best *plausible* reply. A reply counts fully from that probability, and a less likely one in proportion. At 0.15, the engine stops playing unsound moves that lose material to a natural reply, while keeping gambits such as the Belgrade, whose good replies only equalize. Higher values make it more solid, and can push it off such gambits. A higher probability relies more on unlikely refutations being missed.
- **OpponentCautionAhead** (default 0): extra caution where the engine is ahead, OpponentCaution plus this times the engine's expected result there. A winning line's value can't rise much, so without it a line that gives the advantage back for a likely mistake can score the same as one that keeps it. In 200 real-game positions where the engine was ahead, it ended below +1.5 in 15 at 0, 10 at 0.25 and 6 at 0.5. The Belgrade's margin over 5.Nxd4 shrinks with it, and at 1.0 the Belgrade gives way. With OpponentDeepTactics it is less needed; up to 0.25 if the engine still gives back won positions.
- **OpponentDeepTactics** (default 2.0) and **OpponentStockfishPath** (needed for it): a closer look for the opponent where the search relies on it. Once a position where the opponent is to move has **OpponentDeepTacticsVisits** visits (default 500), Stockfish (the executable at OpponentStockfishPath, for example StockfishTrapFinder's; without it the closer look is off) scores all of the opponent's moves there at **OpponentDeepTacticsDepth** (default 10, about 0.25 s a position, so keep OpponentDeepTacticsVisits at 500 or more; 4 is about 15 ms; 8 is more reliable at about 0.03-0.15 s; Stockfish's scores swing at low depths, and at 5-6 can hide an only move, as in LC0_PLAN.md's 21.Qd6 case), in **OpponentDeepTacticsThreads** processes (default 8, one CPU thread each; fewer on a smaller computer), and each move's probability is weighted by exp(−OpponentDeepTactics × loss / 300 cp), its loss against the best move, then renormalized: a blunder within the look's depth is the more likely to be spotted the more it loses, so a plain defense the opponent already favors becomes the clear choice while a rare only move stays unlikely. At 1.0, a 3-pawn loss cuts a move's odds by a factor of e. The depth stands for the opponent's tactical vision: deeper sees more of what a player would, and also some of what a player would miss. With it the engine found Danish-like gambits and attacking lines on its own, and with a horizon of 16 it stopped 8.dxe5 in the Two Knights line (see LC0_PLAN.md).
- **OpponentWideCandidates** (default 0, off), **OpponentWideShare** (default 0.25) and **OpponentWideVisits** (default 2000): a fair hearing below the root for the engine's moves the main network rates low, such as a sacrifice whose attack only shows after some search. At a position where the engine is to move with at least OpponentWideVisits visits, each of its OpponentWideCandidates highest-prior moves first gets OpponentWideShare / OpponentWideCandidates of the position's visits. At 12 it found the knight offer 12...Ng4 one move below the root, which before only OpponentRootCandidates 12 found at the root (see LC0_PLAN.md). A stopgap until the engine's own policy learns which moves trouble human opponents.
- **OpponentHorizon** (default 16; 0 is no limit): how many half-moves ahead the opponent model picks the opponent's moves; beyond that, both sides are searched as in Lc0, so the engine stops counting on the opponent's mistakes long after the opening. With OpponentDeepTactics, 16 dropped the 8.dxe5 gamble while keeping the Belgrade about even with 5.Nxd4; shorter horizons cost the Belgrade (see LC0_PLAN.md).
- **OpponentStockfishFloor** (default off): with OpponentDeepTactics, Stockfish's scores of the opponent's moves also set a floor under the engine's value in the positions they lead to, since the engine plays at least as well as best play there and the opponent can only add mistakes. Meant to make attacks the main network misjudges count at once; in tests it made no measurable difference (see LC0_PLAN.md). (OpponentSoundCaution, which judged refutations by a pessimistic value and undervalued sacrifices, was removed.)
- **OpponentDeepTacticsWait** (default off): with OpponentDeepTactics, the search waits for the closer look at the positions it has queued before searching on. A long search hardly needs it, as Stockfish keeps up; a short one, such as self-play's 800 visits, would end before most of its closer looks arrive. Used for training (training/generate_games.py --search om).
- **OpponentWinValue** (default 1, off): the main network's value from which a position counts as won, with the discount counting plies to where a line is evaluated instead of to the game's predicted end. In tests it pushed the engine off the Belgrade at every setting, and in games against Maia it reached +3 later, not sooner, so leave it at 1 (see LC0_PLAN.md).
- **TreeExportFile** (default empty, off): after each search, writes the search tree to this file as PGN, to see why the engine values a position as it does: the engine's move (the one it plays, then the most searched), the opponent's likely replies as variations with their probabilities, most likely first (a reply of 5% or more is shown even with few visits, as one the closer look made likely late; marked `not searched yet` if it has none), and at the end of each line the search's expected score for the side the engine plays, playing on against the modeled opponent (OpponentDiscount taken out), in the same comment as the move's probability (`8. c3 {30%, Black 91%}` when it plays Black), since a GUI shows only one comment per move. A line ends where the search does or where these limits cut it off, so the score covers the moves below it too. The tree's shape by depth goes to the same path plus `.stats.txt`. **TreeExportDepth** (default 12 half-moves), **TreeExportWidth** (default 4 replies per position) and **TreeExportMinShare** (default 0.01 of the visits) limit its size.
- **OpponentDiscount** (default 0.995; 1 is off): how much less a result counts for each half-move until the game ends, for example 0.99 or 0.995, so the engine prefers winning sooner. Searching a move longer then mostly finds quicker wins instead of pushing every winning move toward 1.0. Evaluations shown are discounted, so they run lower.
- **OpponentEngineMix** (default 0): the share of the main network's move probabilities mixed into Maia's, which makes the modeled opponent stronger, tactics included. 0.25 makes the engine rely less on the opponent missing simple tactics.
- **OpponentTactics** (default 0.25): how much of a quick tactical look the modeled opponent takes. Each of its moves is scored by a depth-1 search with Stockfish 11's classical evaluation, and moves that lose material give up to this share of their probability to the best one. It makes the opponent find clear tactical shots more often, mostly beyond the opening, at about 15 to 25% of the speed. A one-move look can misjudge known opening traps (see [LC0_PLAN.md](LC0_PLAN.md)).
- **OpponentMinProbability** (default 0.002): opponent moves Maia gives less than this probability are taken as never played, so the search goes deeper on the likely ones, like the Stockfish version's threshold.
- **OpponentModelTemperature** (default 1.0): above 1 makes the opponent less predictable, below 1 more. Maia-3 at 1.0 matches how often real players blunder (see [MEASUREMENTS.md](MEASUREMENTS.md)).

The fp16 23M Maia-3 network is the best balance of accuracy and speed: about 35,000 nodes per second, against 69,000 with Maia-1. The first `go` after setting the network takes a few seconds longer, while it loads.

The scores it reports are expected results against the modeled opponent, so they are much higher than plain Lc0's: a strong engine is expected to beat a 1300 player from most positions.

### Building
The source is Lc0 v0.32.1, in [lc0-src/](lc0-src/). To build it on Windows:
1. Install the CUDA toolkit, Visual Studio 2022 or its Build Tools with the C++ workload, and `meson` and `ninja` (`python -m pip install --user meson ninja`).
2. For Maia-3, which runs on ONNX Runtime: unpack ONNX Runtime 1.31's GPU package for CUDA 13 (`onnxruntime-win-x64-gpu_cuda13-1.31.0.zip`) into `lc0/`, and install cuDNN 9, for example with `pip install nvidia-cudnn-cu13` into `.venv` (or set `CUDNN_BIN` to the folder that holds `cudnn64_9.dll`).
3. Run [build_lc0.bat](build_lc0.bat). The result is `lc0-src\build\lc0.exe`, with the ONNX Runtime and cuDNN DLLs copied next to it.

**When to rebuild:** only after a change in [lc0-src/](lc0-src/) or [build_lc0.bat](build_lc0.bat). Changes to the Python tools, the networks or the documents need no rebuild. A new or re-exported network is picked up by setting `OpponentModelWeights` again.

[tools/export_maia3.py](tools/export_maia3.py) exports a Maia-3 checkpoint to ONNX, for example `python tools/export_maia3.py maia3-23m lc0/nets/maia3-23m-fp16.onnx --fp16`.

## The Stockfish Version
To decide which moves shall be kept for the opponent, we use a low-depth search. This search will not be used to find moves our opponent will play, but rather moves the opponent will NOT play. 

For example, if a depth 4 search sees move X as a blunder, our opponent will also likely assume X is a blunder. However, if X turns out to be a good move, and even better, the only good move for our opponent, then we can deduce that our opponent will blunder in this position.

First, our low-depth search will identify the top move in the position. Next, we filter the rest of the moves by comparing their evaluation to the top move. If the move is not within 200 centipawns of the top move, it is discarded.

The design is in [STOCKFISH_DESIGN.md](STOCKFISH_DESIGN.md).

### Model Parameters
Aside from the usual Stockfish parameters, we have a few for controlling our opponent's moves:
- Opponent Depth (default 2, at most 8): Search depth for our opponent. 0 disables opponent modeling. The low-depth search looks at every move, so its cost grows steeply with depth.
- Centipawn Threshold (default 200): The centipawn threshold used for discarding opponent moves.
- Opponent Max Moves (default 256, which is no limit): Of the moves within the threshold, the opponent only plays its best this many, by the low-depth search. Where a position has more legal moves than this, all its moves are decided at once, keeping the best so far with a null-window test against the last of them, much as MultiPV does.
- Opponent Hash (default 16): Size in MB of each thread's filter cache, which remembers the opponent's best score and the verdicts on its moves for each position.

### Building
The engine is based on the Stockfish 19 release (tag `sf_19`). Its source is kept in [stockfish/](stockfish/), and our changes are made directly to it.

To build on Windows, install [MSYS2](https://www.msys2.org/) with the `mingw-w64-x86_64-gcc` and `mingw-w64-x86_64-make` packages, then run [build.bat](build.bat) from the repository root:

```
build.bat
```

This builds `stockfish\src\stockfish.exe` for the current computer's CPU. To build for other machines, pass an architecture, for example `build.bat x86-64-avx2`. MSYS2 is expected in `C:\msys64`; set `MSYS2_ROOT` if it is installed elsewhere. The evaluation network is downloaded on the first build and embedded in the executable, so `stockfish.exe` can be copied anywhere and loaded into any chess GUI that supports UCI.

`build.bat` puts MSYS2's tools at the front of `PATH`. The Makefile runs shell scripts, and they fail under a shell whose path contains a space, such as Git's `C:/Program Files/Git`.

To check the build, run `stockfish.exe bench`. The unmodified source gives `Nodes searched : 2497913`.

### Checking the Opponent Model
The non-standard UCI command `opponentmoves` runs the low-depth search for the side to move in the current position, using the current `Opponent Depth`, `Centipawn Threshold` and `Opponent Max Moves`. It prints every legal move with its exact score and whether it is kept. For example, in the console:

```
position fen rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3
opponentmoves
```

Here only `f3h4`, winning Black's queen, is kept. Every other move is more than 200 centipawns worse.

The command also runs the faster procedure the main search uses, and prints an error if any of its verdicts differ from the exact scores.
