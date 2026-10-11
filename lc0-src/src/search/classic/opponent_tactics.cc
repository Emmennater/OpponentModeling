/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
*/

// Built with Stockfish 11's sources, whose types (Position, Move, Value...)
// are in the global namespace, so this file includes nothing else of Lc0's.

#include "search/classic/opponent_tactics.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "bitboard.h"
#include "endgame.h"
#include "evaluate.h"
#include "movegen.h"
#include "position.h"
#include "thread.h"

// Defined in psqt.cpp, and declared in none of Stockfish 11's headers.
namespace PSQT {
void init();
}

namespace lczero {
namespace classic {
namespace {

constexpr int kMateCp = 10000;
constexpr int kMaxPly = 24;

void InitStockfish() {
  static std::once_flag once;
  std::call_once(once, [] {
    PSQT::init();
    Bitboards::init();
    Position::init();
    Bitbases::init();
    Endgames::init();
  });
}

// Each thread has its own pawn and material hash tables.
Thread& ThisThread() {
  thread_local std::unique_ptr<Thread> thread = std::make_unique<Thread>();
  return *thread;
}

int CaptureOrder(const Position& pos, Move m) {
  return PieceValue[MG][pos.piece_on(to_sq(m))] -
         int(type_of(pos.moved_piece(m)));
}

// Quiescence search: the side to move may stand on the evaluation, or play
// the captures that do not lose material by exchange (all moves in check).
Value QSearch(Position& pos, Value alpha, Value beta, int ply) {
  const bool in_check = pos.checkers();
  Value best = -VALUE_INFINITE;
  if (!in_check) {
    best = Eval::evaluate(pos);
    if (best >= beta || ply >= kMaxPly) return best;
    alpha = std::max(alpha, best);
  } else if (ply >= kMaxPly) {
    return Eval::evaluate(pos);
  }

  ExtMove moves[MAX_MOVES];
  ExtMove* end = in_check ? generate<EVASIONS>(pos, moves)
                          : generate<CAPTURES>(pos, moves);
  for (ExtMove* m = moves; m != end; ++m) m->value = CaptureOrder(pos, m->move);
  std::sort(moves, end, [](const ExtMove& a, const ExtMove& b) {
    return a.value > b.value;
  });

  bool any_legal = false;
  StateInfo st;
  for (ExtMove* m = moves; m != end; ++m) {
    if (!pos.legal(m->move)) continue;
    any_legal = true;
    if (!in_check && !pos.see_ge(m->move)) continue;
    pos.do_move(m->move, st);
    const Value value = -QSearch(pos, -beta, -alpha, ply + 1);
    pos.undo_move(m->move);
    if (value > best) {
      best = value;
      if (value >= beta) return value;
      alpha = std::max(alpha, value);
    }
  }
  if (in_check && !any_legal) return mated_in(ply);
  return best;
}

int ToCentipawns(Value v) {
  if (v >= VALUE_MATE_IN_MAX_PLY) return kMateCp;
  if (v <= VALUE_MATED_IN_MAX_PLY) return -kMateCp;
  return int(v) * 100 / int(PawnValueEg);
}

int Key(int from, int to, char promotion) {
  return (from * 64 + to) * 8 + (promotion == 'q'   ? 1
                                 : promotion == 'r' ? 2
                                 : promotion == 'b' ? 3
                                 : promotion == 'n' ? 4
                                                    : 0);
}

int Key(Move m) {
  constexpr char kPromotions[] = {0, 0, 'n', 'b', 'r', 'q'};
  return Key(int(from_sq(m)), int(to_sq(m)),
             type_of(m) == PROMOTION ? kPromotions[promotion_type(m)] : 0);
}

}  // namespace

std::vector<int> ScoreOpponentMoves(const std::string& fen,
                                    std::span<const TacticsMove> moves) {
  InitStockfish();
  Position pos;
  StateInfo root_state;
  pos.set(fen, false, &root_state, &ThisThread());

  std::unordered_map<int, int> scores;
  StateInfo st;
  for (const ExtMove& m : MoveList<LEGAL>(pos)) {
    pos.do_move(m.move, st);
    const Value value = -QSearch(pos, -VALUE_INFINITE, VALUE_INFINITE, 1);
    pos.undo_move(m.move);
    scores[Key(m.move)] = ToCentipawns(value);
  }

  std::vector<int> result;
  result.reserve(moves.size());
  for (const TacticsMove& m : moves) {
    const auto it = scores.find(Key(m.from, m.to, m.promotion));
    result.push_back(it == scores.end() ? 0 : it->second);
  }
  return result;
}

}  // namespace classic
}  // namespace lczero
