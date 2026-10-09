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

#include "memory.h"
#include "numa.h"
#include "search.h"
#include "types.h"

namespace Stockfish {

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

    // Reset the opponent-model worker's histories, usually before a new game
    void clear();

    // Age the opponent-model hash table. Called once per search by the main thread.
    void new_search();

   private:
    LargePagePtr<Search::Worker> worker;
};

}  // namespace Stockfish

#endif  // #ifndef OPPONENT_MODEL_H_INCLUDED
