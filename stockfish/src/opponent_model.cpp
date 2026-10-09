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

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <memory>
#include <sstream>

#include "history.h"
#include "movegen.h"
#include "movepick.h"
#include "score.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"

namespace Stockfish {

using namespace Search;

OpponentModel::OpponentModel(SharedState&              sharedState,
                             usize                     threadId,
                             usize                     numaThreadId,
                             usize                     numaTotalThreads,
                             NumaReplicatedAccessToken token) :
    filterCache(std::make_unique<FilterEntry[]>(FilterCacheSize)) {

    assert(sharedState.opponentTT == nullptr && sharedState.opponentSharedHistories == nullptr);

    // The thread index only needs to be nonzero: index 0 marks the main thread,
    // which manages time and expects a SearchManager rather than a NullSearchManager.
    worker = make_unique_large_page<Worker>(sharedState, std::make_unique<NullSearchManager>(),
                                            threadId + 1, numaThreadId, numaTotalThreads, token);

    assert(!worker->is_mainthread());

    // search() reads rootMoves[pvIdx].score once rootDepth reaches 16, to decide
    // whether it is looking for a mate. The low-depth search has no root move
    // list of its own, so it gets a single placeholder with a neutral score.
    // It is set up once here, because RootMove allocates memory.
    worker->rootMoves.emplace_back(Move::none());
    worker->rootMoves[0].score = VALUE_ZERO;
    worker->pvIdx              = 0;
    worker->pvLast             = 1;
}

void OpponentModel::clear() {
    worker->clear();
    std::memset(static_cast<void*>(filterCache.get()), 0, FilterCacheSize * sizeof(FilterEntry));
}

void OpponentModel::new_search() { worker->tt.new_search(); }

void OpponentModel::start_search(int depth, int thresholdCp) {
    if (depth != cacheDepth || thresholdCp != cacheThresholdCp)
    {
        std::memset(static_cast<void*>(filterCache.get()), 0,
                    FilterCacheSize * sizeof(FilterEntry));
        cacheDepth       = depth;
        cacheThresholdCp = thresholdCp;
    }

    // Age the move-ordering histories once per main search, as
    // iterative_deepening() does for the main workers at the start of every search
    Worker& w = *worker;

    w.lowPlyHistory.fill(102);

    for (Color c : {WHITE, BLACK})
        for (int i = 0; i < UINT_16_HISTORY_SIZE; i++)
            w.mainHistory[c][i] = w.mainHistory[c][i] * 729 / 1024;
}

// Runs the low-depth search for the side to move in `pos`, which is the
// opponent. Like the main search, it deepens one ply at a time up to `depth`,
// and each iteration searches the moves in the order of the previous
// iteration's scores. Every legal move gets a result from the last iteration
// in `results`: an exact score if it was within `threshold` of the best move
// at the time it was searched, otherwise an upper bound. The best score is
// returned in `bestValue`. Returns false if the search was stopped, in which
// case the results must not be used.
bool OpponentModel::search(Position&                 pos,
                           Depth                     depth,
                           Value                     threshold,
                           const Tablebases::Config& tbConfig,
                           RootResults&              results,
                           Value&                    bestValue) {

    assert(depth > 0 && depth < MAX_PLY);
    assert(threshold >= 0);

    Worker& w = *worker;

    // Set the root fields that search() reads. For the main search they are
    // set by start_searching(), ThreadPool::start_thinking() and
    // iterative_deepening(). The placeholder root move is set up in the
    // constructor, and rootDepth is set by each iteration.
    assert(w.rootMoves.size() == 1 && w.rootMoves[0].score == VALUE_ZERO && w.pvIdx == 0);

    w.accumulatorStack.reset();
    w.nmpMinPly = 0;
    w.selDepth  = 0;
    w.optimism[WHITE] = w.optimism[BLACK] = VALUE_ZERO;
    w.lastIterationIdxPV.clear();
    w.tbConfig = tbConfig;

    // Search stack, set up as in iterative_deepening()
    Stack  stack[MAX_PLY + 10] = {};
    Stack* ss                  = stack + 7;

    for (int i = 7; i > 0; --i)
    {
        (ss - i)->continuationHistory = &w.continuationHistory[0][0][NO_PIECE][0];  // Sentinel
        (ss - i)->continuationCorrectionHistory = &w.continuationCorrectionHistory[NO_PIECE][0];
        (ss - i)->staticEval                    = VALUE_NONE;
    }

    for (int i = 0; i <= MAX_PLY + 2; ++i)
        (ss + i)->ply = i;

    PVMoves rootPv, childPv;
    ss->pv = &rootPv;

    // The root entry is read by the children, for example to decide whether
    // the static evaluation is improving. In check there is no static
    // evaluation, as in search().
    ss->inCheck = pos.checkers();
    ss->ttPv    = true;

    Value unadjustedStaticEval = ss->inCheck ? VALUE_NONE : w.evaluate(pos);
    ss->staticEval             = unadjustedStaticEval;

    auto [ttHit, ttData, ttWriter] = w.tt.probe(pos.key());

    // The first iteration searches the moves in the order the main search would
    RootResult moves[MAX_MOVES];
    usize      moveCount = 0;
    {
        const PieceToHistory* contHist[] = {
          (ss - 1)->continuationHistory, (ss - 2)->continuationHistory,
          (ss - 3)->continuationHistory, (ss - 4)->continuationHistory,
          (ss - 5)->continuationHistory, (ss - 6)->continuationHistory};

        MovePicker mp(pos, ttHit ? ttData.move : Move::none(), depth, &w.mainHistory,
                      &w.lowPlyHistory, &w.captureHistory, contHist, &w.sharedHistory, ss->ply);

        Move move;
        while ((move = mp.next_move()) != Move::none())
            if (pos.legal(move))
                moves[moveCount++] = {move, -VALUE_INFINITE, false};
    }

    assert(moveCount > 0 && moveCount == MoveList<LEGAL>(pos).size());

    StateInfo st;

    for (Depth iterationDepth = 1; iterationDepth <= depth; ++iterationDepth)
    {
        w.rootDepth = iterationDepth;
        bestValue   = -VALUE_INFINITE;

        for (usize i = 0; i < moveCount; ++i)
        {
            RootResult& r = moves[i];

            ss->moveCount = int(i + 1);

            // A move scoring at most alpha is more than `threshold` worse than
            // the best move so far. Since the best score only rises, it would
            // also be discarded at the end, so an upper bound is enough.
            Value alpha = std::max(bestValue - threshold - 1, -VALUE_INFINITE);
            w.rootDelta = VALUE_INFINITE - alpha;

            w.do_move(pos, r.move, st, ss);

            // Cheap null-window test first, then an exact score for moves that
            // pass it. Until a first move has been searched there is nothing
            // to test against.
            Value value = -VALUE_INFINITE;
            if (alpha > -VALUE_INFINITE)
                value =
                  -w.search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, iterationDepth - 1, true);

            if (alpha == -VALUE_INFINITE || value > alpha)
            {
                (ss + 1)->pv = &childPv;
                childPv.clear();
                value =
                  -w.search<PV>(pos, ss + 1, -VALUE_INFINITE, -alpha, iterationDepth - 1, false);
            }

            w.undo_move(pos, r.move);

            // The values of a stopped search are meaningless
            if (w.threads.stop.load(std::memory_order_relaxed))
                return false;

            assert(-VALUE_INFINITE < value && value < VALUE_INFINITE);

            // With an infinite upper bound, a score above alpha is exact. A
            // score at or below alpha, even after passing the test, is an upper bound.
            r.value   = value;
            r.exact   = value > alpha;
            bestValue = std::max(bestValue, value);
        }

        // Order the moves by this iteration's scores: exact scores first, best
        // first, then upper bounds, highest first. The best move is exact,
        // because it scored above the alpha it was searched with.
        std::stable_sort(moves, moves + moveCount, [](const RootResult& a, const RootResult& b) {
            return a.exact != b.exact ? a.exact : a.value > b.value;
        });

        assert(moves[0].exact && moves[0].value == bestValue);
    }

    for (usize i = 0; i < moveCount; ++i)
        results.push_back(moves[i]);

    // The best score is exact, because every move was searched with an
    // infinite upper bound. Storing it orders the best move first next time.
    ttWriter.write(pos.key(), bestValue, true, BOUND_EXACT, depth, moves[0].move,
                   unadjustedStaticEval, w.tt.generation());

    return true;
}

bool OpponentModel::allowed_moves(Worker& owner, Position& pos, AllowedMoves& allowed) {

    assert(owner.opponentDepth > 0 && pos.side_to_move() == owner.opponentColor);
    assert(owner.opponentDepth == cacheDepth && owner.opponentThreshold == cacheThresholdCp);

    MoveList<LEGAL> legalMoves(pos);

    // Nothing to decide with fewer than two moves
    if (legalMoves.size() <= 1)
    {
        for (Move m : legalMoves)
            allowed.push_back(m);
        return true;
    }

    const Key    key   = pos.key();
    FilterEntry& entry = filterCache[key & (FilterCacheSize - 1)];

    if (entry.key != key)
    {
        RootResults results;
        Value       bestValue;
        Value       threshold = UCIEngine::from_cp(owner.opponentThreshold, pos);
        u64         nodesBefore = worker->nodes;

        bool completed =
          search(pos, owner.opponentDepth, threshold, owner.tbConfig, results, bestValue);

        // Count the low-depth search towards the owner's nodes, so node limits
        // and the reported nodes per second include it.
        owner.nodes += u64(worker->nodes) - nodesBefore;

        // Results of a stopped search are never stored
        if (!completed)
            return false;

        std::memset(entry.allowed, 0, sizeof(entry.allowed));

        usize i = 0;
        for (Move m : legalMoves)
        {
            for (const RootResult& r : results)
                if (r.move == m && is_kept(r, bestValue, threshold))
                    entry.allowed[i / 64] |= u64(1) << (i % 64);
            ++i;
        }

        entry.key = key;
    }

    usize i = 0;
    for (Move m : legalMoves)
    {
        if (entry.allowed[i / 64] & (u64(1) << (i % 64)))
            allowed.push_back(m);
        ++i;
    }

    // The best move is always kept
    assert(allowed.size() > 0);

    return true;
}

std::string OpponentModel::describe(Position&                 pos,
                                    int                       depth,
                                    int                       thresholdCp,
                                    const Tablebases::Config& tbConfig) {

    if (depth <= 0)
        return "Opponent modeling is disabled (Opponent Depth is 0).";

    const usize legalCount = MoveList<LEGAL>(pos).size();
    if (legalCount == 0)
        return "No legal moves.";

    RootResults results;
    Value       bestValue;
    Value       threshold   = UCIEngine::from_cp(thresholdCp, pos);
    u64         nodesBefore = worker->nodes;
    TimePoint   startTime   = now();

    if (!search(pos, depth, threshold, tbConfig, results, bestValue))
        return "The search was stopped.";

    const u64       nodes   = u64(worker->nodes) - nodesBefore;
    const TimePoint elapsed = now() - startTime;

    std::stringstream out;
    usize             keptCount = 0;

    out << "Low-depth search for " << (pos.side_to_move() == WHITE ? "white" : "black")
        << " at depth " << depth << ", threshold " << thresholdCp << " cp (" << threshold
        << " internal units)\n";

    // search() returns the moves sorted: exact scores first, best first, then
    // upper bounds, highest first
    for (const RootResult& r : results)
    {
        bool kept = is_kept(r, bestValue, threshold);
        keptCount += kept;

        out << std::left << std::setw(7) << UCIEngine::move(r.move, pos.is_chess960())
            << std::setw(14)
            << (r.exact ? "" : "<= ") + UCIEngine::format_score(Score(r.value, pos))
            << (kept ? "kept" : "discarded") << "\n";
    }

    out << "Kept " << keptCount << " of " << legalCount << " moves (" << nodes << " nodes, "
        << elapsed << " ms)";

    return out.str();
}

}  // namespace Stockfish
