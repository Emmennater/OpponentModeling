/*
  Opponent modeling for Stockfish, a UCI chess playing engine derived from Glaurung 2.1

  This file is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This file is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef OPPONENT_MODEL_H_INCLUDED
#define OPPONENT_MODEL_H_INCLUDED

#include <atomic>
#include <memory>
#include <string>

#include "memory.h"
#include "misc.h"
#include "nnue/network.h"
#include "nnue/nnue_accumulator.h"
#include "numa.h"
#include "position.h"
#include "types.h"

namespace Stockfish {

namespace Search {
class Worker;
}

// Bit set over the legal moves of a position, in the order of MoveList<LEGAL>.
// Legal moves are always generated in the same order for a given position, so
// the bits identify the moves.
struct MoveBits {
    u64 bits[(MAX_MOVES + 63) / 64];

    bool test(usize i) const { return bits[i / 64] & (u64(1) << (i % 64)); }
    void set(usize i) { bits[i / 64] |= u64(1) << (i % 64); }
};

// The opponent model's state for one visit of an opponent node by the main
// search. The moves are decided lazily: only the moves the main search
// actually reaches are tested against the threshold.
struct OpponentNode {
    Move     legalMoves[MAX_MOVES];
    usize    legalCount;
    Key      key;
    Value    bestValue;   // The opponent's best score in the low-depth search
    Value    threshold;   // In internal units
    bool     allAllowed;  // Every legal move is within the threshold
    MoveBits tested;      // The moves whose verdict is known
    MoveBits allowed;     // Of those, the moves within the threshold
};

// OpponentModel decides which moves the opponent would plausibly play, using
// a low-depth search. Each main search worker owns one. A move is allowed if
// the low-depth search scores it at least the opponent's best score minus the
// threshold.
//
// The low-depth search is a plain full-width alpha-beta search followed by a
// search of captures, using Stockfish's move generation and NNUE evaluation.
// It has no hash table, no history tables and no pruning that depends on what
// earlier searches learned, so its verdicts are a function of the position
// alone. Stockfish's own search does not have this property: at low depth its
// pruning depends on its history tables, so the same position could get
// different verdicts.
class OpponentModel {
   public:
    OpponentModel(const LazyNumaReplicatedSystemWide<Eval::NNUE::Network>& network,
                  NumaReplicatedAccessToken                                token,
                  const std::atomic_bool&                                  stop,
                  usize                                                    cacheSizeMb);

    // Clear the filter cache and the NNUE refresh cache, usually before a new game
    void clear();

    // Resize the filter cache. Its contents are lost.
    void resize_cache(usize cacheSizeMb);

    // Called by the owning worker at the start of each search. The filter
    // cache is cleared if the parameters differ from those it was filled with.
    void start_search(int depth, int thresholdCp);

    // Prepares `node` for an opponent node of the main search, with the
    // opponent to move in `pos`: finds the opponent's best score, from the
    // filter cache if possible. Returns false if the search was stopped.
    bool prepare(Search::Worker& owner, Position& pos, OpponentNode& node);

    // Sets `allowed` to whether the opponent would plausibly play the legal
    // move `m` in the position of `node`, testing it if its verdict is not yet
    // known. Returns false if the search was stopped, in which case `allowed`
    // must not be used.
    bool is_allowed(
      Search::Worker& owner, Position& pos, OpponentNode& node, Move m, bool& allowed);

    // Runs the low-depth search on `pos` without the filter cache, computing an
    // exact score for every move, and describes the result. It also checks that
    // the tests used during the main search give the same verdicts. Used by
    // the `opponentmoves` command.
    std::string describe(Position& pos, int depth, int thresholdCp);

   private:
    // Filter cache entry: the opponent's best score, and the verdicts found
    // so far
    struct FilterEntry {
        Key      key;
        Value    bestValue;
        MoveBits tested;
        MoveBits allowed;
    };

    // A root move of the low-depth search
    struct RootMove {
        Move  move;
        Value value;  // Exact score, or -VALUE_INFINITE if not known
        bool  kept;
    };

    static Key cache_key(const Position& pos);

    FilterEntry& cache_entry(Key key) { return filterCache[key & (filterCacheSize - 1)]; }

    // Finds the best score exactly. Moves whose score is found exactly on the
    // way get it in `value`. Returns false if the search was stopped.
    bool find_best(Position& pos, Depth depth, RootMove* moves, usize& moveCount, Value& best);

    // Tests with a null window whether move `m` scores above `alpha`. Returns
    // false if the search was stopped.
    bool test_move(Position& pos, Depth depth, Move m, Value alpha, bool& above);

    // find_best() followed by a test of every move, as used during the main
    // search, but for all moves at once. Used by describe() as a check.
    bool decide(Position& pos, Depth depth, Value threshold, RootMove* moves, usize& moveCount);

    // An exact score for every move. Used by describe().
    bool decide_exact(Position& pos, Depth depth, Value threshold, RootMove* moves, usize& moveCount);

    // Generates the root moves, ordered by a search of captures after each move
    void root_moves(Position& pos, Depth depth, RootMove* moves, usize& moveCount);

    Value alpha_beta(Position& pos, Value alpha, Value beta, Depth depth, int ply);
    Value qsearch(Position& pos, Value alpha, Value beta, int ply);
    Value evaluate(const Position& pos);
    void  do_move(Position& pos, Move m, StateInfo& st);
    void  undo_move(Position& pos, Move m);
    bool  stopped() const { return stop.load(std::memory_order_relaxed); }

    const LazyNumaReplicatedSystemWide<Eval::NNUE::Network>& network;
    NumaReplicatedAccessToken                                numaAccessToken;
    const std::atomic_bool&                                  stop;

    LargePagePtr<Eval::NNUE::AccumulatorStack>  accumulatorStack;
    LargePagePtr<Eval::NNUE::AccumulatorCaches> refreshTable;

    std::unique_ptr<FilterEntry[]> filterCache;
    usize                          filterCacheSize = 0;  // A power of two
    int                            cacheDepth = -1, cacheThresholdCp = -1;

    u64 nodes = 0;
};

}  // namespace Stockfish

#endif  // #ifndef OPPONENT_MODEL_H_INCLUDED
