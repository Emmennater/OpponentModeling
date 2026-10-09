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

The low-depth search reuses Stockfish's own search function instead of a separate implementation. The two searches must not interfere with each other, so all mutable search state is kept separate.

### A Second Worker per Thread
In Stockfish, each search thread runs a `Search::Worker`, which owns the node counter, root moves, search stack, the evaluation network's incremental state and most of the history tables. Two things are shared instead of owned: the hash table, and a set of history tables that all threads on a NUMA node share (continuation, pawn and correction history). Each thread therefore gets a second worker, the opponent-model worker, which runs the unmodified `search()` function against its own state:

| State | Main search | Opponent-model search |
|---|---|---|
| Hash table | The normal one, sized by `Hash` | Its own small table |
| Per-thread history tables | Main worker's | Its own |
| Shared history tables | Shared by the main workers | A separate set, shared by the opponent-model workers |
| Evaluation network state | Main worker's | Its own, rebuilt from the position at the start of each low-depth search |
| Search stack | Main worker's | Its own |
| `Position` | Shared | Shared; every move is undone, so the position is returned unchanged |
| Stop flag | Shared | Shared, so stopping the engine also stops the low-depth search |

The opponent-model worker never runs as the main thread, so it does not manage time. It also has a flag that disables move filtering inside it, so it never calls itself recursively. It is created alongside the main worker, on the same thread, so it uses memory on the same NUMA node.

`search()` reads some values that Stockfish's normal root code sets up, so the low-depth search sets them itself: the root move list, the root depth, the root window size, the optimism values and the sentinel entries at the start of the search stack.

### Filtering the Moves
The low-depth search uses its own root loop. Stockfish's normal root loop only finds the best move, but we need every move within the threshold of the best. To get these, the lower bound of the search window (alpha) is set to `best - threshold` and raised as better moves are found:

1. Order the opponent's legal moves, with the best move from the opponent-model hash table first.
2. For each move, run a cheap null-window search to test whether it scores above `best - threshold`.
3. If it fails low, it is discarded. Its true score is at most `best - threshold`, and `best` only rises, so it can never come back.
4. If it passes, re-search it with an open window to get its exact score and keep it. If that score is a new best, discard any kept moves that are now more than `threshold` below it.

The best move is always kept, so the result is never empty. Searching the best move first makes every later test search cheaper.

### Hooking into the Main Search
Only two changes are made to `search()`:
- At nodes where the opponent is to move, the allowed moves are computed just before the moves loop, after the `moves_loop` label so that nodes in check are covered too. Nodes that end earlier (hash table hits, razoring, futility pruning, null-move pruning) never pay for a low-depth search.
- In the moves loop, disallowed moves are skipped before they are counted. Late-move reductions and move-count pruning therefore treat the allowed moves as the whole move list.

The result for each position is cached by its hash key. The cache saves time when a position is searched again. It also keeps the main search consistent: without it, a position could get a different list of allowed moves later in the same search, because the opponent-model hash table keeps changing. Results from a search that was stopped part-way are never cached, and the cache is cleared on `ucinewgame` or when an opponent parameter changes.

### Known Limitations
- ProbCut tries captures before the moves loop, so it may use a capture the opponent would not play. This errs in the opponent's favor and is left unchanged for now.
- Every opponent node that reaches the moves loop runs a full low-depth search, even close to the leaves. If this is too slow, filtering could be skipped when the remaining depth is low.

## Stockfish Version and Building
The engine is based on the Stockfish 19 release (tag `sf_19`). Its source is kept in [stockfish/](stockfish/), and our changes are made directly to it.

To build on Windows with MSYS2's MinGW toolchain:

```
cd stockfish/src
mingw32-make -j build ARCH=x86-64-bmi2 COMP=mingw
```

`ARCH=native` fails when the shell's path contains a space (such as `C:/Program Files/Git`), so the architecture is given explicitly. Run `bash ../scripts/get_native_properties.sh` to find the right value for your CPU. The evaluation network is downloaded automatically on the first build.

To check the build, run `./stockfish.exe bench`. The unmodified source gives `Nodes searched : 2497913`.

## Model Parameters
Aside from the usual Stockfish parameters, we have a few for controlling our opponent's moves:
- Opponent Depth: Search depth for our opponent.
- Centipawn Threshold: The centipawn threshold used for discarding opponent moves.