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

#include <memory>
#include <string>

#include "memory.h"
#include "misc.h"
#include "numa.h"
#include "position.h"
#include "search.h"
#include "syzygy/tbprobe.h"
#include "types.h"

namespace Stockfish {

// The opponent moves that the main search is allowed to search at a node
using AllowedMoves = ValueList<Move, MAX_MOVES>;

// OpponentModel decides which moves the opponent would plausibly play, using
// a low-depth search. Each main search worker owns one. The search runs in a
// separate Search::Worker, the opponent-model worker, which has its own hash
// table, histories and NNUE accumulators, so it never touches the state of
// the main search.
class OpponentModel {
   public:
    // The hash table and shared histories in `sharedState` must be the
    // opponent-model ones, and its own opponent-model fields must be null, so
    // the opponent-model worker gets no opponent model of its own. The thread
    // and NUMA indices are those of the owning worker.
    OpponentModel(Search::SharedState&      sharedState,
                  usize                     threadId,
                  usize                     numaThreadId,
                  usize                     numaTotalThreads,
                  NumaReplicatedAccessToken token);

    // Reset the opponent-model worker's histories and the filter cache,
    // usually before a new game.
    void clear();

    // Age the opponent-model hash table. Called once per search by the main thread.
    void new_search();

    // Called by the owning worker at the start of each search. The filter
    // cache is cleared if the parameters differ from those it was filled with.
    void start_search(int depth, int thresholdCp);

    // Fills `allowed` with the moves the opponent, to move in `pos`, would
    // plausibly play, using the owning worker's parameters. Returns false if
    // the search was stopped, in which case `allowed` must not be used.
    bool allowed_moves(Search::Worker& owner, Position& pos, AllowedMoves& allowed);

    // Runs the low-depth search on `pos` without the filter cache, and
    // describes the result. Used by the `opponentmoves` command.
    std::string describe(Position&                 pos,
                         int                       depth,
                         int                       thresholdCp,
                         const Tablebases::Config& tbConfig);

   private:
    // Result of the low-depth search for one root move
    struct RootResult {
        Move  move;
        Value value;  // The exact score if `exact`, otherwise an upper bound
        bool  exact;
    };
    using RootResults = ValueList<RootResult, MAX_MOVES>;

    // Filter cache entry. Bit i of `allowed` is set if the i-th move of
    // MoveList<LEGAL> is allowed. Legal moves are always generated in the
    // same order for a given position, so the bits identify the moves.
    struct FilterEntry {
        Key key;
        u64 allowed[(MAX_MOVES + 63) / 64];
    };

    static constexpr usize FilterCacheSize = usize(1) << 16;

    static bool is_kept(const RootResult& r, Value best, Value threshold) {
        return r.exact && r.value >= best - threshold;
    }

    bool search(Position&                 pos,
                Depth                     depth,
                Value                     threshold,
                const Tablebases::Config& tbConfig,
                RootResults&              results,
                Value&                    bestValue);

    LargePagePtr<Search::Worker>   worker;
    std::unique_ptr<FilterEntry[]> filterCache;
    int                            cacheDepth = -1, cacheThresholdCp = -1;
};

}  // namespace Stockfish

#endif  // #ifndef OPPONENT_MODEL_H_INCLUDED
