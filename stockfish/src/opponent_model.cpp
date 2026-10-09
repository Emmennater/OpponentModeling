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
#include <cassert>
#include <cstring>
#include <iomanip>
#include <sstream>

#include "evaluate.h"
#include "movegen.h"
#include "score.h"
#include "search.h"
#include "uci.h"

namespace Stockfish {

namespace {

// Move ordering only affects the speed of the low-depth search, never its
// verdicts: the search is full-width, so it finds the same scores in any
// order. Captures and promotions come first, most valuable victim first, then
// least valuable attacker first.
int order_score(const Position& pos, Move m) {
    if (!pos.capture_stage(m))
        return 0;

    int score = 1 << 20;
    score += 16 * PieceValue[pos.piece_on(m.to_sq())];  // No piece there for en passant
    if (m.type_of() == PROMOTION)
        score += 16 * PieceValue[make_piece(WHITE, m.promotion_type())];

    return score - int(type_of(pos.moved_piece(m)));
}

// Fills `list` with the legal moves of the given type, in search order.
// Returns the number of moves.
template<GenType Type>
int ordered_moves(const Position& pos, ExtMove* list) {
    int n = 0;
    for (Move m : MoveList<Type>(pos))
        if (Type == LEGAL || pos.legal(m))
        {
            list[n]       = m;
            list[n].value = order_score(pos, m);
            ++n;
        }

    std::stable_sort(list, list + n,
                     [](const ExtMove& a, const ExtMove& b) { return a.value > b.value; });
    return n;
}

// Index of the legal move `m` in the node's list of legal moves
usize index_of(const OpponentNode& node, Move m) {
    const Move* it = std::find(node.legalMoves, node.legalMoves + node.legalCount, m);
    assert(it != node.legalMoves + node.legalCount);
    return usize(it - node.legalMoves);
}

}  // namespace

OpponentModel::OpponentModel(const LazyNumaReplicatedSystemWide<Eval::NNUE::Network>& net,
                             NumaReplicatedAccessToken                                token,
                             const std::atomic_bool&                                  stopFlag,
                             usize                                                    cacheSizeMb) :
    network(net),
    numaAccessToken(token),
    stop(stopFlag),
    accumulatorStack(make_unique_large_page<Eval::NNUE::AccumulatorStack>()),
    refreshTable(make_unique_large_page<Eval::NNUE::AccumulatorCaches>(network[token])) {

    resize_cache(cacheSizeMb);
}

void OpponentModel::clear() {
    refreshTable->clear(network[numaAccessToken]);
    std::memset(static_cast<void*>(filterCache.get()), 0, filterCacheSize * sizeof(FilterEntry));
}

void OpponentModel::resize_cache(usize cacheSizeMb) {
    const usize entries = std::max<usize>(1, cacheSizeMb * 1024 * 1024 / sizeof(FilterEntry));

    // A power of two, so an entry can be found by masking the key
    filterCacheSize = 1;
    while (filterCacheSize * 2 <= entries)
        filterCacheSize *= 2;

    filterCache = std::make_unique<FilterEntry[]>(filterCacheSize);  // Zero-initialized
}

void OpponentModel::start_search(int depth, int thresholdCp) {
    if (depth != cacheDepth || thresholdCp != cacheThresholdCp)
    {
        std::memset(static_cast<void*>(filterCache.get()), 0,
                    filterCacheSize * sizeof(FilterEntry));
        cacheDepth       = depth;
        cacheThresholdCp = thresholdCp;
    }
}

// The verdict depends on the exact 50-move counter, because the evaluation
// is scaled down as it grows. pos.key() only includes it in steps of 8 plies.
Key OpponentModel::cache_key(const Position& pos) {
    return pos.key() ^ (Key(pos.rule50_count()) + 1) * 0x9E3779B97F4A7C15ULL;
}

Value OpponentModel::evaluate(const Position& pos) {
    return Eval::evaluate(network[numaAccessToken], pos, *accumulatorStack, *refreshTable, 0);
}

void OpponentModel::do_move(Position& pos, Move m, StateInfo& st) {
    auto& dirties = accumulatorStack->push();
    pos.do_move(m, st, pos.gives_check(m), dirties, nullptr, nullptr);
    ++nodes;
}

void OpponentModel::undo_move(Position& pos, Move m) {
    pos.undo_move(m);
    accumulatorStack->pop();
}

// Full-width, fail-soft alpha-beta search. There is no draw detection except
// stalemate, so the result depends only on the position, not on how it was
// reached.
Value OpponentModel::alpha_beta(Position& pos, Value alpha, Value beta, Depth depth, int ply) {

    assert(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);

    if (depth <= 0)
        return qsearch(pos, alpha, beta, ply);

    if (stopped())
        return VALUE_ZERO;

    ExtMove   list[MAX_MOVES];
    const int moveCount = ordered_moves<LEGAL>(pos, list);

    if (moveCount == 0)
        return pos.checkers() ? mated_in(ply) : VALUE_DRAW;

    StateInfo st;
    Value     bestValue = -VALUE_INFINITE;

    for (int i = 0; i < moveCount; ++i)
    {
        do_move(pos, list[i], st);
        Value value = -alpha_beta(pos, -beta, -alpha, depth - 1, ply + 1);
        undo_move(pos, list[i]);

        if (value > bestValue)
        {
            bestValue = value;
            if (value > alpha)
            {
                alpha = value;
                if (value >= beta)
                    break;
            }
        }
    }

    return bestValue;
}

// Search of captures, to resolve exchanges before evaluating. Out of check,
// the side to move may stand pat, and captures that lose material by static
// exchange evaluation are not tried. In check, every evasion is tried.
Value OpponentModel::qsearch(Position& pos, Value alpha, Value beta, int ply) {

    assert(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);

    if (stopped())
        return VALUE_ZERO;

    const bool inCheck = pos.checkers();

    if (ply >= MAX_PLY - 1)
        return inCheck ? VALUE_DRAW : evaluate(pos);

    ExtMove list[MAX_MOVES];
    int     moveCount;
    Value   bestValue = -VALUE_INFINITE;

    if (inCheck)
        moveCount = ordered_moves<EVASIONS>(pos, list);
    else
    {
        bestValue = evaluate(pos);
        if (bestValue >= beta)
            return bestValue;

        alpha     = std::max(alpha, bestValue);
        moveCount = ordered_moves<CAPTURES>(pos, list);
    }

    StateInfo st;

    for (int i = 0; i < moveCount; ++i)
    {
        if (!inCheck && !pos.see_ge(list[i], 0))
            continue;

        do_move(pos, list[i], st);
        Value value = -qsearch(pos, -beta, -alpha, ply + 1);
        undo_move(pos, list[i]);

        if (value > bestValue)
        {
            bestValue = value;
            if (value > alpha)
            {
                alpha = value;
                if (value >= beta)
                    break;
            }
        }
    }

    if (inCheck && bestValue == -VALUE_INFINITE)
        return mated_in(ply);

    return bestValue;
}

// Fills `moves` with the legal root moves. For a search deeper than one ply,
// they are ordered by a search of captures after each move, best first, so
// that the best move is likely searched first. At depth 1 that would be the
// whole search, done twice.
void OpponentModel::root_moves(Position& pos, Depth depth, RootMove* moves, usize& moveCount) {

    ExtMove   list[MAX_MOVES];
    const int n = ordered_moves<LEGAL>(pos, list);

    if (depth > 1)
    {
        StateInfo st;
        for (int i = 0; i < n; ++i)
        {
            do_move(pos, list[i], st);
            list[i].value = -qsearch(pos, -VALUE_INFINITE, VALUE_INFINITE, 1);
            undo_move(pos, list[i]);
        }

        std::stable_sort(list, list + n,
                         [](const ExtMove& a, const ExtMove& b) { return a.value > b.value; });
    }

    for (int i = 0; i < n; ++i)
        moves[i] = {list[i], -VALUE_INFINITE, false};

    moveCount = usize(n);
}

// Finds the opponent's best score exactly. The first move is searched with an
// open window. Each later move is tested with a null window against the best
// score so far, and re-searched with an open window only if it is better.
// Moves whose score is found exactly get it in `value`. Returns false if the
// search was stopped.
bool OpponentModel::find_best(
  Position& pos, Depth depth, RootMove* moves, usize& moveCount, Value& bestValue) {

    assert(depth > 0);

    accumulatorStack->reset();
    root_moves(pos, depth, moves, moveCount);

    StateInfo st;
    bestValue = -VALUE_INFINITE;

    for (usize i = 0; i < moveCount; ++i)
    {
        RootMove& rm = moves[i];

        do_move(pos, rm.move, st);

        Value value;
        if (i == 0)
            value = -alpha_beta(pos, -VALUE_INFINITE, VALUE_INFINITE, depth - 1, 1);
        else
        {
            value = -alpha_beta(pos, -(bestValue + 1), -bestValue, depth - 1, 1);
            if (value > bestValue)
                value = -alpha_beta(pos, -VALUE_INFINITE, -bestValue, depth - 1, 1);
        }

        undo_move(pos, rm.move);

        if (stopped())
            return false;

        // Searched with an infinite upper bound and scoring above its lower
        // bound, so the score is exact
        if (i == 0 || value > bestValue)
        {
            rm.value  = value;
            bestValue = value;
        }
    }

    return true;
}

// Tests with a null window whether move `m` scores above `alpha`. The result
// is exact: a full-width search finds the same answer in any order. Returns
// false if the search was stopped.
bool OpponentModel::test_move(Position& pos, Depth depth, Move m, Value alpha, bool& above) {

    assert(depth > 0 && -VALUE_INFINITE <= alpha && alpha < VALUE_INFINITE);

    accumulatorStack->reset();

    StateInfo st;
    do_move(pos, m, st);
    Value value = -alpha_beta(pos, -(alpha + 1), -alpha, depth - 1, 1);
    undo_move(pos, m);

    if (stopped())
        return false;

    above = value > alpha;
    return true;
}

// A move is kept if it scores at least `best - threshold`. Moves whose score
// find_best() found exactly are compared directly; the others are tested.
bool OpponentModel::decide(
  Position& pos, Depth depth, Value threshold, RootMove* moves, usize& moveCount) {

    assert(threshold >= 0);

    Value bestValue;
    if (!find_best(pos, depth, moves, moveCount, bestValue))
        return false;

    const Value cutoff = bestValue - threshold;

    for (usize i = 0; i < moveCount; ++i)
    {
        RootMove& rm = moves[i];

        if (rm.value != -VALUE_INFINITE)
            rm.kept = rm.value >= cutoff;

        // When the best score is a forced loss close to -VALUE_INFINITE,
        // every move is within the threshold
        else if (cutoff - 1 < -VALUE_INFINITE)
            rm.kept = true;

        else if (!test_move(pos, depth, rm.move, cutoff - 1, rm.kept))
            return false;
    }

    return true;
}

// The slower search used by describe(): an exact score for every move.
// Returns false if the search was stopped.
bool OpponentModel::decide_exact(
  Position& pos, Depth depth, Value threshold, RootMove* moves, usize& moveCount) {

    assert(depth > 0 && threshold >= 0);

    accumulatorStack->reset();
    root_moves(pos, depth, moves, moveCount);

    StateInfo st;
    Value     bestValue = -VALUE_INFINITE;

    for (usize i = 0; i < moveCount; ++i)
    {
        do_move(pos, moves[i].move, st);
        moves[i].value = -alpha_beta(pos, -VALUE_INFINITE, VALUE_INFINITE, depth - 1, 1);
        undo_move(pos, moves[i].move);

        if (stopped())
            return false;

        bestValue = std::max(bestValue, moves[i].value);
    }

    for (usize i = 0; i < moveCount; ++i)
        moves[i].kept = moves[i].value >= bestValue - threshold;

    return true;
}

bool OpponentModel::prepare(Search::Worker& owner, Position& pos, OpponentNode& node) {

    assert(owner.opponentDepth > 0 && pos.side_to_move() == owner.opponentColor);
    assert(owner.opponentDepth == cacheDepth && owner.opponentThreshold == cacheThresholdCp);

    node.legalCount = usize(generate<LEGAL>(pos, node.legalMoves) - node.legalMoves);

    // Nothing to decide with fewer than two moves
    node.allAllowed = node.legalCount <= 1;
    if (node.allAllowed)
        return true;

    node.key       = cache_key(pos);
    node.threshold = UCIEngine::from_cp(owner.opponentThreshold, pos);

    FilterEntry& entry = cache_entry(node.key);

    if (entry.key == node.key)
    {
        node.bestValue = entry.bestValue;
        node.tested    = entry.tested;
        node.allowed   = entry.allowed;
    }
    else
    {
        RootMove moves[MAX_MOVES];
        usize    moveCount;
        u64      nodesBefore = nodes;

        bool completed = find_best(pos, owner.opponentDepth, moves, moveCount, node.bestValue);

        // Count the low-depth search towards the owner's nodes, so node limits
        // and the reported nodes per second include it
        owner.nodes += nodes - nodesBefore;

        // Results of a stopped search are never stored
        if (!completed)
            return false;

        assert(moveCount == node.legalCount);

        // Moves whose score find_best() found exactly need no test
        node.tested = node.allowed = MoveBits{};

        for (usize j = 0; j < moveCount; ++j)
            if (moves[j].value != -VALUE_INFINITE)
            {
                const usize i = index_of(node, moves[j].move);
                node.tested.set(i);
                if (moves[j].value >= node.bestValue - node.threshold)
                    node.allowed.set(i);
            }

        entry = {node.key, node.bestValue, node.tested, node.allowed};
    }

    // When the best score is a forced loss close to -VALUE_INFINITE, every
    // move is within the threshold
    node.allAllowed = node.bestValue - node.threshold - 1 < -VALUE_INFINITE;

    return true;
}

bool OpponentModel::is_allowed(
  Search::Worker& owner, Position& pos, OpponentNode& node, Move m, bool& allowed) {

    if (node.allAllowed)
    {
        allowed = true;
        return true;
    }

    assert(cache_key(pos) == node.key);

    const usize i = index_of(node, m);

    if (!node.tested.test(i))
    {
        bool above;
        u64  nodesBefore = nodes;

        bool completed = test_move(pos, owner.opponentDepth, m,
                                   node.bestValue - node.threshold - 1, above);

        owner.nodes += nodes - nodesBefore;

        if (!completed)
            return false;

        node.tested.set(i);
        if (above)
            node.allowed.set(i);

        // Store the verdict. While searching earlier moves of this node, the
        // main search may have replaced the cache entry with another position.
        FilterEntry& entry = cache_entry(node.key);

        if (entry.key != node.key)
            entry = {node.key, node.bestValue, node.tested, node.allowed};
        else
        {
            entry.tested.set(i);
            if (above)
                entry.allowed.set(i);
        }
    }

    allowed = node.allowed.test(i);
    return true;
}

std::string OpponentModel::describe(Position& pos, int depth, int thresholdCp) {

    if (depth <= 0)
        return "Opponent modeling is disabled (Opponent Depth is 0).";

    const usize legalCount = MoveList<LEGAL>(pos).size();
    if (legalCount == 0)
        return "No legal moves.";

    const Value threshold = UCIEngine::from_cp(thresholdCp, pos);

    RootMove exact[MAX_MOVES], fast[MAX_MOVES];
    usize    exactCount, fastCount;

    u64       nodesBefore = nodes;
    TimePoint startTime   = now();

    if (!decide_exact(pos, depth, threshold, exact, exactCount))
        return "The search was stopped.";

    const u64       exactNodes = nodes - nodesBefore;
    const TimePoint exactTime  = now() - startTime;

    nodesBefore = nodes;
    startTime   = now();

    if (!decide(pos, depth, threshold, fast, fastCount))
        return "The search was stopped.";

    const u64       fastNodes = nodes - nodesBefore;
    const TimePoint fastTime  = now() - startTime;

    // The fast search, used during the main search, must keep the same moves
    bool agree = exactCount == fastCount;
    for (usize i = 0; i < exactCount && agree; ++i)
    {
        const RootMove* f = std::find_if(fast, fast + fastCount, [&](const RootMove& rm) {
            return rm.move == exact[i].move;
        });
        agree = f != fast + fastCount && f->kept == exact[i].kept;
    }

    std::stable_sort(exact, exact + exactCount,
                     [](const RootMove& a, const RootMove& b) { return a.value > b.value; });

    std::stringstream out;
    usize             keptCount = 0;

    out << "Low-depth search for " << (pos.side_to_move() == WHITE ? "white" : "black")
        << " at depth " << depth << ", threshold " << thresholdCp << " cp (" << threshold
        << " internal units)\n";

    for (usize i = 0; i < exactCount; ++i)
    {
        keptCount += exact[i].kept;

        out << std::left << std::setw(7) << UCIEngine::move(exact[i].move, pos.is_chess960())
            << std::setw(14) << UCIEngine::format_score(Score(exact[i].value, pos))
            << (exact[i].kept ? "kept" : "discarded") << "\n";
    }

    out << "Kept " << keptCount << " of " << legalCount << " moves (exact scores: " << exactNodes
        << " nodes, " << exactTime << " ms; fast search: " << fastNodes << " nodes, " << fastTime
        << " ms)";

    if (!agree)
        out << "\nERROR: the fast search used during the main search keeps different moves";

    return out.str();
}

}  // namespace Stockfish
