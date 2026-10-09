# Implementation Plan
This document describes how opponent modeling is added to Stockfish 19: which parts of the engine change, what each change does, and why the rest of the engine keeps working. The overall idea is in [README.md](README.md).

File and line references point to the unmodified `sf_19` source, as committed in `edcae62`. Line numbers will drift once changes are made.

## Overview
The opponent's moves are restricted at every node of the main search where the opponent is to move. A low-depth search decides which moves are allowed. It is a small, separate search written for this purpose. It uses Stockfish's move generation and evaluation network, but not Stockfish's search function. Its verdict on each move depends only on the position (see [section 9](#9-history-why-not-stockfishs-own-search) for why).

| File | Change |
|---|---|
| `opponent_model.h`, `opponent_model.cpp` (new) | The low-depth search, the decisions on which moves are allowed, and the filter cache |
| [search.h](stockfish/src/search.h), [search.cpp](stockfish/src/search.cpp) | An `OpponentModel` per worker, and the two hooks in `search()` |
| [engine.h](stockfish/src/engine.h), [engine.cpp](stockfish/src/engine.cpp) | UCI options, resizing the filter caches, clearing the main hash table when the model changes |
| [thread.cpp](stockfish/src/thread.cpp) | Copy the parameters into each worker at the start of a search |
| [uci.h](stockfish/src/uci.h), [uci.cpp](stockfish/src/uci.cpp) | Centipawn-to-internal-units conversion, and the `opponentmoves` debug command |
| [Makefile](stockfish/src/Makefile) | Add `opponent_model.cpp` and `opponent_model.h` |

## 1. Parameters
Three UCI options are added in the `Engine` constructor ([engine.cpp:107](stockfish/src/engine.cpp#L107), next to `Skill Level`):

| Option | Default | Range | Meaning |
|---|---|---|---|
| `Opponent Depth` | 2 | 0–8 | Depth of the low-depth search. 0 disables opponent modeling. |
| `Centipawn Threshold` | 200 | 0–2000 | Moves more than this many centipawns worse than the opponent's best move are discarded. |
| `Opponent Hash` | 16 | 1–`MaxHashMB` | Size in MB of each thread's filter cache. |

The low-depth search looks at every move, so its cost grows steeply with depth. That's why the depth is limited to 8 and defaults to 2.

`Opponent Hash` works like the existing `Hash` option ([engine.cpp:89-93](stockfish/src/engine.cpp#L89-L93)): changing it resizes the caches through a callback. Each thread resizes its own cache on its own thread, so the memory is on that thread's NUMA node.

Looking up an option by name is too slow to do at every node, so `Opponent Depth` and `Centipawn Threshold` are copied into each worker once per search. This happens in the per-thread setup in `ThreadPool::start_thinking` ([thread.cpp:337-349](stockfish/src/thread.cpp#L337-L349)), which already copies the limits and root position into every worker. The opponent's color is set there too, as the side not to move at the root.

**Centipawns are not Stockfish's internal units.** Stockfish reports scores in centipawns using a scale that depends on the material on the board ([uci.cpp:585-594](stockfish/src/uci.cpp#L585-L594)). At full material, 100 centipawns is about 283 internal units. A new function `UCIEngine::from_cp(int cp, const Position& pos)` does the reverse conversion. The threshold is converted for each position, so "200 centipawns" means the same thing it would in Stockfish's own output for that position.

## 2. The Opponent Model
Each `Search::Worker` owns an `OpponentModel`, created in the `Worker` constructor ([search.cpp:164-184](stockfish/src/search.cpp#L164-L184)). Workers are created on their own thread so their memory lands on the right NUMA node ([thread.cpp:69-77](stockfish/src/thread.cpp#L69-L77)), and the opponent model with them. `OpponentModel` is declared a `friend` of `Worker` ([search.h:425-426](stockfish/src/search.h#L425-L426)), so it can read the parameters and add to the node count.

The opponent model needs very little state of its own:

| State | Main search | Low-depth search | Why they can't interfere |
|---|---|---|---|
| Evaluation network state (accumulator stack and refresh cache) | Main worker's | The opponent model's own | Separate objects; the low-depth search resets its accumulator stack each time it starts |
| Hash table, history tables | Main worker's and shared | None | The low-depth search has none |
| `Position` | Shared object | Shared object | Every `do_move` in the low-depth search is paired with an `undo_move`, so the position is returned exactly as it was |
| Stop flag (`threads.stop`) | Shared | Shared, read only | Intended: stopping the engine also stops the low-depth search |
| Node count | Main worker's | The opponent model's own, added to the main worker's | See below |
| Network weights | Shared | Shared | Read only |

The low-depth search's nodes are added to the main worker's node count, so node limits (`go nodes`) and the reported nodes per second include all the work done. This is the one channel through which the low-depth search affects the main search beyond the moves it allows. Stockfish uses the low bits of the node count as a cheap source of randomness, to vary draw scores slightly ([search.cpp:134](stockfish/src/search.cpp#L134)) and to break ties between equal moves ([search.cpp:1511](stockfish/src/search.cpp#L1511)). Adding nodes changes those choices, so the main search takes different but equally valid paths. A test with the model's decisions computed but not applied, and its nodes not counted, gave results identical to plain Stockfish. That confirms nothing else leaks.

## 3. The Low-Depth Search
The low-depth search is a plain, full-width, fail-soft alpha-beta search to the opponent depth, followed by a search of captures:
- **At depth > 0:** every legal move is searched. A position with no legal moves is mate or stalemate.
- **The capture search (depth 0):** out of check, the side to move may "stand pat" on the static evaluation, and only captures and queen promotions that don't lose material by static exchange evaluation are tried. In check, every evasion is tried, so mates are found.
- **Evaluation:** Stockfish's own `Eval::evaluate` with the NNUE network, with optimism 0.
- **Move ordering:** captures first, most valuable victim first, then least valuable attacker. At the root, for depth > 1, moves are ordered by the capture search after each move, best first.

### Why its verdicts depend only on the position
- **No hash table and no history tables.** Nothing is carried from one search to the next, except the network's refresh cache, which only makes evaluation faster and never changes its result.
- **No pruning except alpha-beta.** In a full-width search, alpha-beta returns the exact minimax value whenever that value lies inside the search window, whatever order the moves are tried in. Outside the window it returns a bound on the right side of it. So "is this move above the cutoff?" always gets the same answer.
- **No draw detection except stalemate.** Repetitions and the 50-move rule depend on how a position was reached, so they are ignored.
- **The 50-move counter is part of the key.** The evaluation scales scores down as the counter grows ([evaluate.cpp:63](stockfish/src/evaluate.cpp#L63)). `pos.key()` only includes the counter in steps of 8 plies once it reaches 14 ([position.h:322](stockfish/src/position.h#L322)), so the filter cache's key mixes in the exact count.

Mate scores are measured from the root of the low-depth search, so they don't depend on the main search either.

## 4. Deciding Which Moves Are Allowed
A move is allowed if the low-depth search scores it at least `best - threshold`, where `best` is the opponent's best score. The decision is split in two, so that work is only done for moves the main search actually reaches:

1. **Best score** (`find_best`), once per opponent position. The first move is searched with an open window. Each later move is tested with a null window against the best score so far, and re-searched with an open window only if it might be better. This is principal variation search at the root. Moves whose score is found exactly on the way, including the best move, get their verdict immediately.
2. **Per-move test** (`test_move`), only when the main search is about to search a move whose verdict isn't known yet. A null-window search answers "does this move score above `best - threshold - 1`?".

If the opponent has only one legal move, nothing is searched. If the best score is a forced loss so close to `-VALUE_INFINITE` that `best - threshold - 1` would be out of range, every move is allowed. A move that loses as quickly as the best one can't be distinguished from it anyway.

### Why the result is correct
- **The best score is exact.** The first move is searched with an open window. A later move replaces it only if it scored above the old best in an open-upper-bound search, which returns an exact value.
- **Each test is exact.** It returns whether the move's true depth-limited value is above the cutoff, independent of move order, for the reasons in section 3.
- **The allowed set is never empty.** The best move is always within the threshold of itself, so the main search always has at least one opponent move when one is legal.

The `opponentmoves` command checks this. It computes an exact score for every move, then runs `find_best` and tests every move, and reports an error if the two disagree.

## 5. Filter Cache
Each worker's `OpponentModel` keeps a filter cache, sized by `Opponent Hash` (a power of two of 80-byte entries; 16 MB gives 131,072). An entry holds:
- the position key, with the exact 50-move count mixed in;
- the opponent's best score;
- a 256-bit mask of the moves whose verdict is known, and a 256-bit mask of those that are allowed. Bit *i* refers to the *i*-th move of `MoveList<LEGAL>`. Legal move generation always lists a position's moves in the same order, so the bits identify the moves.

The best score is stored when an opponent position is first prepared, and each verdict as it is found. While the main search works through an opponent position's moves, its recursion may overwrite that position's cache entry. The position's state is also kept in a local `OpponentNode` in the `search()` frame, so nothing is lost; the entry is written again with the next verdict.

Rules:
- Results from a stopped search are never stored.
- The cache is cleared in `Worker::clear()` ([search.cpp:690](stockfish/src/search.cpp#L690)), which runs on `ucinewgame`.
- Each cache records the depth and threshold it was filled with. At the start of a search, it is cleared if either has changed.

Since every verdict depends only on the position, the cache only saves time. A recomputed verdict is always the same.

## 6. Changes to `search()`
### Preparing an opponent node
Right after Step 13, the small ProbCut check ([search.cpp:1100-1104](stockfish/src/search.cpp#L1100-L1104)), and before the `MovePicker` is created ([search.cpp:1111](stockfish/src/search.cpp#L1111)):

```cpp
OpponentNode opponentNode;
const bool   restrictMoves = opponentDepth > 0 && us == opponentColor;
if (restrictMoves && !opponentModel->prepare(*this, pos, opponentNode))
    return VALUE_ZERO;  // The search was stopped
```

This point comes after the `moves_loop` label ([search.cpp:1098](stockfish/src/search.cpp#L1098)), so nodes in check, which jump straight to the label, are covered. It also comes after every early exit in `search()`: hash table cutoffs, razoring, futility pruning, null-move pruning, ProbCut and the small ProbCut. Nodes that end early never run a low-depth search.

When the search is stopped, it returns the same kind of throwaway value Stockfish returns at Step 2 ([search.cpp:788-790](stockfish/src/search.cpp#L788-L790)). The main search discards results from a stopped iteration.

The root is never an opponent node, because the side to move at the root is by definition not the opponent.

### Skipping disallowed moves
After Step 15, Stockfish's pruning at shallow depths ([search.cpp:1164-1232](stockfish/src/search.cpp#L1164-L1232)), and before Step 16, singular extensions:

```cpp
if (restrictMoves)
{
    bool allowed;
    if (!opponentModel->is_allowed(*this, pos, opponentNode, move, allowed))
        return VALUE_ZERO;  // The search was stopped
    if (!allowed)
    {
        ss->moveCount = --moveCount;
        continue;
    }
}
```

The check comes after Stockfish's own pruning, so moves that would be pruned anyway are never tested. Near the leaves, Stockfish prunes most quiet moves this way. Testing them first made the benchmark about 1.5 times slower.

### Why the rest of `search()` keeps working
From the main search's point of view, the game tree is the normal one with some of the opponent's moves removed, and which moves are removed depends only on the position. Alpha-beta search and the hash table are correct on any such tree, so the remaining question is whether Stockfish's heuristics still make sense on it:

- **Hash table cutoffs, razoring, futility pruning, null-move pruning** run before the hook, using values that were themselves computed on the restricted tree. They estimate the same thing as before, the value of the node.
- **Move-count pruning and late-move reductions** use `moveCount`. A disallowed move is uncounted again when it is skipped, so they see the allowed moves plus the moves Stockfish pruned at Step 15. Stockfish counts pruned moves too ([search.cpp:1137](stockfish/src/search.cpp#L1137) comes before Step 15). A disallowed move that Stockfish prunes stays counted, as it would in plain Stockfish.
- **Futility pruning of quiet moves** can raise the node's best value to the pruned move's futility estimate ([search.cpp:1218-1224](stockfish/src/search.cpp#L1218-L1224)). For a disallowed move pruned this way, that estimate errs in the opponent's favor. That is a conservative error, so it is accepted.
- **Singular extensions** search the same node again with the hash table move excluded. The second visit finds the best score in the filter cache. If the excluded move was the only allowed move, the moves loop finds no moves, and `search()` already returns alpha for that case ([search.cpp:1565-1566](stockfish/src/search.cpp#L1565-L1566)), which is exactly the singular case.
- **Mate and stalemate detection** ([search.cpp:1554-1566](stockfish/src/search.cpp#L1554-L1566)) relies on `moveCount == 0` meaning there are no legal moves. The allowed set is never empty when a legal move exists, so this still holds, and so does the assertion at [search.cpp:1556](stockfish/src/search.cpp#L1556).
- **History updates** only see moves that were searched, so disallowed moves are never rewarded or penalized.
- **ProbCut** ([search.cpp:1054-1096](stockfish/src/search.cpp#L1054-L1096)) runs before the hook and may use a capture the opponent would not play. That errs in the opponent's favor, and it is left unchanged, as noted in the README.
- **Quiescence search** is unchanged, as decided in the README.

### Clearing the main hash table when the model changes
Values in the main hash table depend on which side is the opponent, and on the depth and threshold. `Engine::go` ([engine.cpp:153](stockfish/src/engine.cpp#L153)) remembers the color to move and the two parameters from the previous search. If any of them has changed, it clears the main hash table before starting. In a normal game the engine always plays the same color, so this only happens when analyzing positions for both sides or when the options change.

## 7. Debug Command
The UCI command `opponentmoves` runs the low-depth search for the side to move in the current position, without the filter cache. It prints every legal move with its exact score and whether it is kept, the threshold in internal units, and the nodes and time taken. It also runs the faster decision procedure used during the main search and prints an error if any verdict differs. It runs on the main search thread through `ThreadPool::run_on_thread`, so it uses the thread's large stack and NUMA binding. It is added next to the other non-standard commands such as `d` and `eval` ([uci.cpp:148-165](stockfish/src/uci.cpp#L148-L165)).

## 8. Testing
With the model enabled, the engine searches a different tree, so Stockfish's own checks no longer have a known right answer. The `bench` node count changes, and the signature test in [tests/signature.sh](stockfish/tests/signature.sh) fails by design. Most of the testing is therefore manual. The automated checks that still apply are used where they can be:

1. **Disabled model:** with `Opponent Depth` set to 0, `bench` must report exactly `Nodes searched : 2497913`, the same as unmodified `sf_19`. This shows the changes do nothing when disabled.
2. **Move generation:** perft (as in [tests/perft.sh](stockfish/tests/perft.sh)) must give the standard counts, for example 4,865,609 at depth 5 from the start position.
3. **Assertions:** build with `debug=yes` and run `bench` with the model enabled, with one thread and with four. Stockfish's assertions check, among other things, search windows, depths, and the mate/stalemate condition above.
4. **Reproducibility:** [tests/reprosearch.sh](tests/reprosearch.sh), a bash version of [Stockfish's test](stockfish/tests/reprosearch.sh), checks that a search repeated after `ucinewgame` searches exactly the same number of nodes. It takes the opponent depth as an argument.
5. **Verdicts depend only on the position:** run `opponentmoves` on a position after main searches of different depths in the same process. The output must be identical each time.
6. **Fast and exact searches agree:** `opponentmoves` reports an error if they disagree. Run it on chosen positions at several opponent depths.
7. **Filter behavior (manual):** check the kept and discarded moves by hand on chosen positions. These include one with a single hard-to-find good move, positions in check, positions with a forced mate for either side, and a position with only one legal move.
8. **Loading in a GUI (manual):** load the engine in a chess GUI, check that the three options appear and can be changed, and play a few games with the model enabled.
9. **Speed:** compare the time to reach a given depth with the model enabled and disabled, at several opponent depths.
10. **Strength (later):** play matches against an opponent such as fixed-depth Stockfish or Maia, using a tool like `fastchess` or `cutechess-cli`.

## 9. History: Why Not Stockfish's Own Search
The first version ran Stockfish's own `search()` as the low-depth search, in a second `Search::Worker` per thread with its own hash table and history tables. It produced wrong traps. In one game the engine claimed +7.43, built on the opponent being allowed exactly one move after `9.Qc1` in this line:

`1.d4 d5 2.Bf4 Nf6 3.h3 c5 4.dxc5 Nc6 5.Nf3 Qa5+ 6.Qd2 Qxc5 7.e3 e6 8.Be2 Bd6`

Run on its own, the same depth-2 search allowed 34 moves in that position. The difference came from the opponent-model worker's state:

| State of the opponent-model worker | `...Nb4` scored | Moves kept |
|---|---|---|
| Fresh, or after a depth-1 main search | −31 cp | 34 of 47 |
| After a main search to depth 2–5 | +279 cp | 1 of 47 |
| After a main search to depth 6–10 | about −15 cp | 34 of 47 |

At low depth, Stockfish prunes most quiet moves, and which ones it looks at depends on its history tables. Those change with every search. When White's defense wasn't among the moves looked at, `...Nb4` looked winning, and the filter cache then held that verdict for the whole search. The opponent model has to give each position the same verdict every time, so it was replaced with the search in section 3.

The replacement is slower. Stockfish's selective search took a few hundred nodes per opponent position at depth 2. The full-width search takes a few hundred to about 2,000 to decide every move, of which finding the best score is roughly a third. Deciding each move only when the main search reaches it, and only after Stockfish's own pruning, brought the benchmark at the default depth from 98 s down to 67 s, about 27 times plain Stockfish.

## Possible Later Changes
- A hash table inside the low-depth search, keyed by position and depth, storing only true bounds of the depth-limited value. It would keep the verdicts the same while sharing work, for example between the best-score search and the per-move tests, and between the low-depth searches of sibling positions, which often reach the same positions by transposition.
- Skipping the low-depth search when the main search's remaining depth is low. This would make the model apply only away from the leaves. A quick measurement showed modest gains.
- Restricting the opponent's moves in ProbCut and quiescence search.
- A minimum number of allowed moves, so the opponent is never forced into a single move by a threshold measured from an overestimated best move.
