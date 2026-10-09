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

#include "opponent_model.h"

#include <cassert>
#include <memory>

#include "tt.h"

namespace Stockfish {

OpponentModel::OpponentModel(Search::SharedState&      sharedState,
                             usize                     threadId,
                             usize                     numaThreadId,
                             usize                     numaTotalThreads,
                             NumaReplicatedAccessToken token) {

    assert(sharedState.opponentTT == nullptr && sharedState.opponentSharedHistories == nullptr);

    // The thread index only needs to be nonzero: index 0 marks the main thread,
    // which manages time and expects a SearchManager rather than a NullSearchManager.
    worker = make_unique_large_page<Search::Worker>(
      sharedState, std::make_unique<Search::NullSearchManager>(), threadId + 1, numaThreadId,
      numaTotalThreads, token);

    assert(!worker->is_mainthread());
}

void OpponentModel::clear() { worker->clear(); }

void OpponentModel::new_search() { worker->tt.new_search(); }

}  // namespace Stockfish
