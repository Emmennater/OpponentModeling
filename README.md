# Opponent Modeling Using Stockfish
The problem with modern chess engines is that they assume the opponent will always play the best move. The goal of this project is to modify the search algorithm used by stockfish to handicap the opponent's moves. The problem is that changing the algorithm significantly might disrupt existing optimizations, leaving us to design our own engine. To get around this, the idea is to restrict the move list to a set of moves the opponent would reasonably play.

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
It needs two networks: a main network for its own play, and a Maia network for the opponent. These UCI options control the opponent model:
- **OpponentModelWeights:** the Maia network file. It can be a Maia-3 network exported to ONNX, for example `lc0/nets/maia3-23m-fp16.onnx`, or a Maia-1 network, for example `lc0/nets/maia-1300.pb.gz`. Empty turns opponent modeling off.
- **OpponentModelRating** and **OpponentModelOwnRating** (default 1300 each), for Maia-3 only: the opponent's rating, and the rating of the side the engine plays.
- **OpponentModelTemperature** (default 1.0): above 1 makes the opponent less predictable, below 1 more. Maia-3 at 1.0 matches how often real players blunder (see [MEASUREMENTS.md](MEASUREMENTS.md)).

The fp16 23M Maia-3 network is the best balance of accuracy and speed: about 24,000 nodes per second, against 69,000 with Maia-1. The first `go` after setting the network takes a few seconds longer, while it loads.

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
The non-standard UCI command `opponentmoves` runs the low-depth search for the side to move in the current position, using the current `Opponent Depth` and `Centipawn Threshold`. It prints every legal move with its exact score and whether it is kept. For example, in the console:

```
position fen rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3
opponentmoves
```

Here only `f3h4`, winning Black's queen, is kept. Every other move is more than 200 centipawns worse.

The command also runs the faster procedure the main search uses, and prints an error if any of its verdicts differ from the exact scores.
