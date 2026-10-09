# Implementation Plan
This document describes how opponent modeling is added to Stockfish 19: which parts of the engine change, what each change does, and why the rest of the engine keeps working. The overall idea and design are in [README.md](README.md).

File and line references point to the unmodified `sf_19` source, as committed in `edcae62`. Line numbers will drift once changes are made.

## Overview
The opponent's moves are restricted at every node of the main search where the opponent is to move. A low-depth search decides which moves are kept. It runs in a second `Search::Worker` per thread, the opponent-model worker, which runs Stockfish's unmodified `search()` function against its own copy of every piece of mutable search state.

| File | Change |
|---|---|
| `opponent_model.h`, `opponent_model.cpp` (new) | The opponent-model worker, the root loop of the low-depth search, and the filter cache |
| [search.h](stockfish/src/search.h) | Opponent-model fields in `SharedState` and `Worker`, and a `friend` declaration |
| [search.cpp](stockfish/src/search.cpp) | Create the opponent-model worker, compute allowed moves before the moves loop, skip disallowed moves |
| [engine.h](stockfish/src/engine.h), [engine.cpp](stockfish/src/engine.cpp) | UCI options, the opponent-model hash table and shared histories, clearing the main hash table when the model changes |
| [thread.cpp](stockfish/src/thread.cpp) | Create the opponent-model shared histories, copy the parameters into each worker at the start of a search |
| [uci.h](stockfish/src/uci.h), [uci.cpp](stockfish/src/uci.cpp) | Centipawn-to-internal-units conversion, and an `opponentmoves` debug command |
| [Makefile](stockfish/src/Makefile) | Add `opponent_model.cpp` and `opponent_model.h` |

## 1. Parameters
Two UCI options are added in the `Engine` constructor ([engine.cpp:107](stockfish/src/engine.cpp#L107), next to `Skill Level`):

| Option | Default | Range | Meaning |
|---|---|---|---|
| `Opponent Depth` | 4 | 0–20 | Depth of the low-depth search. 0 disables opponent modeling. |
| `Centipawn Threshold` | 200 | 0–2000 | Moves more than this many centipawns worse than the best move are discarded. |

Looking up an option by name is too slow to do at every node, so both values are copied into each worker once per search. This happens in the per-thread setup in `ThreadPool::start_thinking` ([thread.cpp:337-349](stockfish/src/thread.cpp#L337-L349)), which already copies the limits and root position into every worker. The opponent's color is set there too, as the side not to move at the root.

**Centipawns are not Stockfish's internal units.** Stockfish reports scores in centipawns using a scale that depends on the material on the board ([uci.cpp:585-594](stockfish/src/uci.cpp#L585-L594)). At full material, 100 centipawns is about 304 internal units. A new function `UCIEngine::from_cp(int cp, const Position& pos)` is added next to `to_cp` to do the reverse conversion. The threshold is converted at the root of each low-depth search, so "200 centipawns" means the same thing it would in Stockfish's own output for that position.

## 2. Engine-Level State
The opponent-model workers need their own hash table and their own set of shared history tables. Both are owned by `Engine`, next to the existing ones ([engine.h:130-136](stockfish/src/engine.h#L130-L136)):

- **`opponentTT`:** a `TranspositionTable` with a fixed size of 16 MB, shared by all opponent-model workers, just as `tt` is shared by all main workers. It is resized in `Engine::set_tt_size` ([engine.cpp:257](stockfish/src/engine.cpp#L257)) and cleared in `Engine::search_clear` ([engine.cpp:161](stockfish/src/engine.cpp#L161)). Its age (`new_search()`) is advanced by the main thread at the start of each search, where the main table's is.
- **`opponentSharedHists`:** a `std::map<NumaIndex, SharedHistories>`, built in `ThreadPool::set` with the same per-NUMA-node thread counts as the main one ([thread.cpp:208-220](stockfish/src/thread.cpp#L208-L220)).

`SharedState` ([search.h:194-211](stockfish/src/search.h#L194-L211)) gets two optional pointers to these. The `SharedState` built for an opponent-model worker leaves them null, which is how a worker knows not to create an opponent model of its own.

**Why:** in `sf_19`, continuation, pawn and correction history are not owned by each worker. All threads on a NUMA node share one set ([history.h:206-257](stockfish/src/history.h#L206-L257)). Correction history adjusts the static evaluation ([search.cpp:85-131](stockfish/src/search.cpp#L85-L131)), so if the low-depth search wrote to the main set, it would change the main search's evaluations. A separate set prevents that.

## 3. The Opponent-Model Worker
A new class `OpponentModel` holds the opponent-model worker and the filter cache. Each main worker owns one through a `std::unique_ptr`, created in the `Worker` constructor ([search.cpp:164-184](stockfish/src/search.cpp#L164-L184)).

The opponent-model worker is an ordinary `Search::Worker`, built with:
- a `SharedState` that points at `opponentTT` and `opponentSharedHists`, with the same options, thread pool and network as the main worker,
- a `NullSearchManager`, the manager helper threads already use, so it never checks the clock,
- a nonzero thread index, because `is_mainthread()` is `threadIdx == 0` ([search.h:343](stockfish/src/search.h#L343)) and the main thread's code would treat its manager as a `SearchManager`,
- the main worker's NUMA indices and access token.

Main workers are created on their own thread so their memory lands on the right NUMA node ([thread.cpp:69-77](stockfish/src/thread.cpp#L69-L77)). Creating the opponent-model worker inside the main worker's constructor puts it on the same thread, and so on the same node.

`OpponentModel` is declared a `friend` of `Worker` ([search.h:425-426](stockfish/src/search.h#L425-L426)), so it can call the private `search()` and set the root fields described below. The opponent-model worker has no `OpponentModel` of its own, so the low-depth search can never call itself.

### What is shared and what is not

| State | Main search | Low-depth search | Why they can't interfere |
|---|---|---|---|
| Hash table | `tt` | `opponentTT` | Separate tables |
| Per-thread histories | Main worker's | Opponent-model worker's | Separate workers |
| Shared histories | `sharedHists` | `opponentSharedHists` | Separate sets |
| Evaluation network state (`accumulatorStack`, `refreshTable`) | Main worker's | Opponent-model worker's | Separate workers; reset at the start of each low-depth search |
| Search stack | Main worker's | A local array in the root loop | Separate memory |
| `Position` | Shared object | Shared object | Every `do_move` in the low-depth search is paired with an `undo_move`, so the position is returned exactly as it was |
| Stop flag (`threads.stop`) | Shared | Shared, read only | Intended: stopping the engine also stops the low-depth search |
| Options, network weights | Shared | Shared | Read only |

## 4. The Low-Depth Search
`OpponentModel::filter(Position& pos, ...)` runs the low-depth search for the opponent to move in `pos` and returns the allowed moves, or reports that it was stopped.

### Setup
`search()` reads several fields that Stockfish's normal root code sets up (`start_searching`, `iterative_deepening` and `start_thinking`). The root loop sets them itself on the opponent-model worker:

- **Network state:** `accumulatorStack.reset()`, as `start_searching` does ([search.cpp:193](stockfish/src/search.cpp#L193)). The first evaluation rebuilds the state from the position, using the worker's refresh cache.
- **Search stack:** a local `Stack` array with the same sentinel entries `iterative_deepening` sets up ([search.cpp:289-303](stockfish/src/search.cpp#L289-L303)).
- **Root entry of the stack:** `inCheck`, and the root's static evaluation. Children compare their own evaluation against it to set the `improving` and `opponentWorsening` flags ([search.cpp:863-864](stockfish/src/search.cpp#L863-L864)).
- **`rootDepth`:** the opponent depth. It is used in the singular extension margins.
- **`rootMoves` and `pvIdx`:** a single entry with score 0, and `pvIdx = 0`. Every call to `search()` reads `rootMoves[pvIdx].score` once `rootDepth >= 16` ([search.cpp:727](stockfish/src/search.cpp#L727)), so the list must not be empty.
- **`rootDelta`:** the width of the current root window. The reduction formula divides by it ([search.cpp:1885-1888](stockfish/src/search.cpp#L1885-L1888)).
- **`optimism`:** 0 for both colors. The main search derives it from the previous iteration's score, which the low-depth search does not have.
- **`nmpMinPly`, `selDepth`:** 0. **`lastIterationIdxPV`:** cleared. **`tbConfig`:** copied from the main worker, so tablebases are used the same way.

### Root loop
The legal moves are ordered by a `MovePicker`, built the same way as in `search()` ([search.cpp:1111-1112](stockfish/src/search.cpp#L1111-L1112)) but with the opponent-model worker's histories and hash table move. The lower bound of the root window (alpha) is kept at `best - threshold` and raised as better moves are found:

```
best = -infinity, kept = {}
for each legal move m, in MovePicker order:
    a = max(best - threshold, -VALUE_INFINITE)
    rootDelta = VALUE_INFINITE - a
    do_move(m)
    v = -search<NonPV>(pos, ss + 1, -(a + 1), -a, depth - 1, cutNode = true)   // test: is v > a?
    if v > a:
        v = -search<PV>(pos, ss + 1, -VALUE_INFINITE, -a, depth - 1, false)    // exact score
    undo_move(m)
    if stopped: return "stopped"
    if v <= a: continue                       // discard
    add (m, v) to kept
    if v > best:
        best = v
        remove from kept every move with value <= best - threshold
return the moves in kept
```

This is principal variation search with a moving lower bound. Most moves need only the cheap test search. A move is re-searched with an open window only when it might be kept.

### Why the result is correct
- **A discarded move would also be discarded at the end.** A move fails the test only if its true score is at most `a = best - threshold`. Since `best` only rises, the move is at least `threshold` below the final best too.
- **Kept moves can be filtered again without searching them again.** Each kept move has an exact score, so when `best` rises, the new cutoff can be applied to the stored values directly.
- **The result is never empty.** The first move is searched with `a = -VALUE_INFINITE`, so it is kept, and the best move is never removed. The main search therefore always finds at least one opponent move when there is a legal one.
- **Mate scores are handled.** `a` is clamped to `-VALUE_INFINITE`, because `best - threshold` can go below it when the opponent is being mated. If the opponent has a forced mate, only the mating moves come within the threshold, so only they are kept.

### Other details
- If the opponent has only one legal move, no low-depth search is run.
- The opponent-model worker's node count is added to the main worker's after each low-depth search. Node limits (`go nodes`) and the reported nodes per second then include all the work done.
- The low-depth search does a single pass at the full opponent depth, with no iterative deepening. Its hash table supplies the first move to try in positions it has seen before.

## 5. Filter Cache
Each main worker has its own filter cache, held in its `OpponentModel`. It is a fixed-size table of 2^16 entries. Each entry holds a 64-bit position key and a 256-bit mask. Bit *i* of the mask says whether the *i*-th move of `MoveList<LEGAL>` is allowed. Legal move generation always lists a position's moves in the same order, so the mask fits in 32 bytes no matter how many moves are kept. In quiet positions most moves are often within 200 centipawns, so storing the moves themselves would take far more space. The table takes about 2.5 MB per thread.

On a lookup, the legal moves are generated once and the allowed ones are copied into a list on the stack, which the moves loop then checks against.

The cache does two jobs:
- **Speed:** positions recur often in a search. The singular extension check also searches the same node a second time ([search.cpp:1253-1255](stockfish/src/search.cpp#L1253-L1255)).
- **Consistency:** the opponent-model hash table changes as it is used, so a second low-depth search of the same position can give a slightly different answer. The cache makes each position's allowed moves fixed within a search, which keeps the values in the main hash table consistent with each other.

Rules:
- Results from a stopped search are never stored.
- The cache is cleared in `Worker::clear()` ([search.cpp:690](stockfish/src/search.cpp#L690)), which runs on `ucinewgame`.
- Each cache records the depth and threshold it was filled with. At the start of a search, it is cleared if either has changed.
- If an entry is overwritten and its position comes up again, the recomputed answer may differ slightly. Stockfish already tolerates similar inconsistencies, such as racy hash table writes and positions reached by different move orders, so this is accepted.

## 6. Changes to `search()`
### Computing the allowed moves
Right after Step 13, the small ProbCut check ([search.cpp:1100-1104](stockfish/src/search.cpp#L1100-L1104)), and before the `MovePicker` is created ([search.cpp:1111](stockfish/src/search.cpp#L1111)):

```cpp
// Restrict the opponent to the moves a low-depth search finds plausible
AllowedMoves allowed;
const bool   restrictMoves = opponentModel && opponentDepth > 0 && us == opponentColor;
if (restrictMoves && !opponentModel->allowed_moves(pos, allowed))
    return VALUE_ZERO;  // Search was stopped
```

This point comes after the `moves_loop` label ([search.cpp:1098](stockfish/src/search.cpp#L1098)), so nodes in check, which jump straight to the label, are covered. It also comes after every early exit in `search()`: hash table cutoffs, razoring, futility pruning, null-move pruning, ProbCut and the small ProbCut. Nodes that end early never run a low-depth search.

When the search is stopped, it returns the same kind of throwaway value Stockfish returns at Step 2 ([search.cpp:788-790](stockfish/src/search.cpp#L788-L790)). The main search discards results from a stopped iteration.

The root is never an opponent node, because the side to move at the root is by definition not the opponent.

### Skipping disallowed moves
In the moves loop, right after the legality check ([search.cpp:1128-1129](stockfish/src/search.cpp#L1128-L1129)):

```cpp
if (restrictMoves && !allowed.contains(move))
    continue;
```

### Why the rest of `search()` keeps working
From the main search's point of view, the game tree is the normal one with some of the opponent's moves removed. Within a search, which moves are removed depends only on the position, because of the filter cache. Alpha-beta search and the hash table are correct on any such tree, so the remaining question is whether Stockfish's heuristics still make sense on it:

- **Hash table cutoffs, razoring, futility pruning, null-move pruning** run before the hook, using values that were themselves computed on the restricted tree. They estimate the same thing as before, the value of the node.
- **Move-count pruning and late-move reductions** use `moveCount`, which is only incremented after the new check ([search.cpp:1137](stockfish/src/search.cpp#L1137)). They treat the allowed moves as the whole move list.
- **Singular extensions** search the same node again with the hash table move excluded. The second visit finds the allowed moves in the filter cache. If the excluded move was the only allowed move, the moves loop finds no moves, and `search()` already returns alpha for that case ([search.cpp:1565-1566](stockfish/src/search.cpp#L1565-L1566)), which is exactly the singular case.
- **Mate and stalemate detection** ([search.cpp:1554-1566](stockfish/src/search.cpp#L1554-L1566)) relies on `moveCount == 0` meaning there are no legal moves. The allowed list is never empty when a legal move exists, so this still holds, and so does the assertion at [search.cpp:1556](stockfish/src/search.cpp#L1556).
- **History updates** only see moves that were searched, so disallowed moves are never rewarded or penalized.
- **ProbCut** ([search.cpp:1054-1096](stockfish/src/search.cpp#L1054-L1096)) runs before the hook and may use a capture the opponent would not play. That errs in the opponent's favor, and it is left unchanged, as noted in the README.
- **Quiescence search** is unchanged, as decided in the README.

### Clearing the main hash table when the model changes
Values in the main hash table depend on which side is the opponent, and on the depth and threshold. `Engine::go` ([engine.cpp:153](stockfish/src/engine.cpp#L153)) remembers the color to move and the two parameters from the previous search. If any of them has changed, it clears the main hash table before starting. In a normal game the engine always plays the same color, so this only happens when analyzing positions for both sides or when the options change. The opponent-model hash table does not need clearing: it holds ordinary Stockfish search results, which don't depend on either.

## 7. Debug Command
A new UCI command, `opponentmoves`, runs the low-depth search for the side to move in the current position and prints every legal move with its score, or its upper bound if it was discarded, and whether it was kept. It runs on the main search thread through `ThreadPool::run_on_thread`, so it uses the thread's large stack and NUMA binding. It is added next to the other non-standard commands such as `d` and `eval` ([uci.cpp:148-165](stockfish/src/uci.cpp#L148-L165)). This makes it possible to check the filter on chosen positions without running a full search.

## 8. Testing
1. **Disabled model:** with `Opponent Depth` set to 0, `bench` must report exactly `Nodes searched : 2497913`, the same as unmodified `sf_19`. This shows the changes do nothing when disabled.
2. **Assertions:** build with `debug=yes` and run `bench` with the model enabled. Stockfish's assertions check, among other things, search windows, depths, and the mate/stalemate condition above.
3. **Filter behavior:** run `opponentmoves` on chosen positions. These include positions with one hard-to-find good move, positions in check, positions with a forced mate for either side, and positions with a single legal move.
4. **Speed:** compare nodes per second and time to reach a given depth with the model enabled and disabled, at several opponent depths.
5. **Strength (later):** play matches against an opponent such as fixed-depth Stockfish or Maia, using a tool like `fastchess` or `cutechess-cli`.

## 9. Implementation Steps
Each step is a separate commit that builds and passes the checks listed for it.

1. **Plumbing:** UCI options, `from_cp`, the opponent-model hash table and shared histories, `SharedState` fields, and the `OpponentModel` class with its worker but no search yet. `bench` still gives 2497913.
2. **Low-depth search:** the root loop, the filter cache and the `opponentmoves` command. Check the filter on chosen positions.
3. **Search integration:** the hook and skip in `search()`, node counting, and clearing the main hash table when the model changes. Check `bench` with the model disabled, then a debug build with it enabled.
4. **Measurement and documentation:** speed measurements, and README updates with the results and any changes made along the way.

## Possible Later Changes
- A UCI option for the opponent-model hash table size.
- Skipping the low-depth search when the remaining depth is low, if it turns out to be too slow near the leaves.
- Restricting the opponent's moves in ProbCut and quiescence search.
- Iterative deepening in the low-depth search, if its move ordering turns out to matter.
