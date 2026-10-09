# Opponent Modeling in Stockfish: Design and First Results
How the Stockfish version restricts the opponent's moves, and how it performed. The idea is in the [README](README.md), and the step-by-step implementation, with references into the Stockfish source, is in [PLAN.md](PLAN.md).

## Integrating with Stockfish
Since we are only changing the set of legal moves for our opponent, the rest of the optimizations should still work. It is worth noting that the set of moves will never be empty, unless the original set of moves is also empty.

It is worth noting that the performance of stockfish will degrade dramatically as the opponent depth is increased. The usual move generation process only takes a few nanoseconds, but it will take much longer with additional searches. On the plus side, it will reduce the number of positions considered by the opponent. 

### Quiescence Search
For now, quiescence search is left unchanged: the opponent may still play any capture there. Only the main search restricts the opponent's moves.

## Design of the Low-Depth Search
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
