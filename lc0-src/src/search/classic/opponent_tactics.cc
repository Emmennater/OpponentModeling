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
#include <deque>
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

namespace {

// An alpha-beta search without a hash table, for a few half-moves: captures
// first (most valuable victim, least valuable attacker), then killer moves,
// then checks. Checks extend the search, up to kMaxPly. It stops early when
// `stop` becomes true.
class DeepSearch {
 public:
  DeepSearch(Position& pos, const std::atomic<bool>* stop)
      : pos_(pos), stop_(stop) {}

  bool aborted() const { return aborted_; }

  Value Search(int depth, Value alpha, Value beta, int ply) {
    if (aborted_) return VALUE_ZERO;
    if ((++nodes_ & 1023) == 0 && stop_ &&
        stop_->load(std::memory_order_relaxed)) {
      aborted_ = true;
      return VALUE_ZERO;
    }
    const bool in_check = pos_.checkers();
    if (in_check) ++depth;
    if (depth <= 0 || ply >= kMaxPly) return QSearch(pos_, alpha, beta, ply);

    ExtMove moves[MAX_MOVES];
    ExtMove* end = generate<LEGAL>(pos_, moves);
    if (end == moves) return in_check ? mated_in(ply) : VALUE_DRAW;
    Order(moves, end, ply);
    Value best = -VALUE_INFINITE;
    StateInfo st;
    for (ExtMove* m = moves; m != end; ++m) {
      pos_.do_move(m->move, st);
      const Value value = -Search(depth - 1, -beta, -alpha, ply + 1);
      pos_.undo_move(m->move);
      if (aborted_) return VALUE_ZERO;
      if (value > best) {
        best = value;
        if (value > alpha) alpha = value;
        if (value >= beta) {
          if (!pos_.capture(m->move) && killers_[ply][0] != m->move) {
            killers_[ply][1] = killers_[ply][0];
            killers_[ply][0] = m->move;
          }
          break;
        }
      }
    }
    return best;
  }

  void Order(ExtMove* begin, ExtMove* end, int ply) {
    for (ExtMove* m = begin; m != end; ++m) {
      m->value = pos_.capture(m->move)         ? 100000 + CaptureOrder(pos_, m->move)
                 : m->move == killers_[ply][0] ? 90000
                 : m->move == killers_[ply][1] ? 80000
                 : pos_.gives_check(m->move)   ? 70000
                                               : 0;
    }
    std::stable_sort(begin, end, [](const ExtMove& a, const ExtMove& b) {
      return a.value > b.value;
    });
  }

 private:
  Position& pos_;
  const std::atomic<bool>* stop_;
  uint64_t nodes_ = 0;
  bool aborted_ = false;
  Move killers_[kMaxPly + 1][2] = {};
};

}  // namespace

std::vector<int> ScoreOpponentMovesDeep(const std::string& fen,
                                        std::span<const TacticsMove> moves,
                                        int depth,
                                        const std::atomic<bool>* stop) {
  InitStockfish();
  Position pos;
  StateInfo root_state;
  pos.set(fen, false, &root_state, &ThisThread());
  DeepSearch search(pos, stop);

  ExtMove list[MAX_MOVES];
  ExtMove* end = generate<LEGAL>(pos, list);
  search.Order(list, end, 0);
  // Each move is searched in a window from 8 pawns below the best so far to
  // just above it, and again without the upper bound if it beats the best.
  const int margin = 8 * int(PawnValueEg);
  Value best = -VALUE_INFINITE;
  std::unordered_map<int, int> scores;
  StateInfo st;
  for (ExtMove* m = list; m != end; ++m) {
    pos.do_move(m->move, st);
    Value value;
    if (best == -VALUE_INFINITE) {
      value = -search.Search(depth, -VALUE_INFINITE, VALUE_INFINITE, 1);
    } else {
      const Value floor = Value(std::max(int(best) - margin, -int(VALUE_INFINITE) + 1));
      value = -search.Search(depth, -Value(best + 1), -floor, 1);
      if (value > best) value = -search.Search(depth, -VALUE_INFINITE, -floor, 1);
    }
    pos.undo_move(m->move);
    if (search.aborted()) return {};
    best = std::max(best, value);
    scores[Key(m->move)] = ToCentipawns(value);
  }

  std::vector<int> result;
  result.reserve(moves.size());
  for (const TacticsMove& m : moves) {
    const auto it = scores.find(Key(m.from, m.to, m.promotion));
    result.push_back(it == scores.end() ? -kMateCp : it->second);
  }
  return result;
}

struct SanWriter::Impl {
  Position pos;
  // A deque, so that the states' addresses, which the position keeps, stay put.
  std::deque<StateInfo> states;
  std::vector<Move> played;

  Move Find(const TacticsMove& move) {
    const int key = Key(move.from, move.to, move.promotion);
    for (const ExtMove& m : MoveList<LEGAL>(pos)) {
      if (Key(m.move) == key) return m.move;
    }
    return MOVE_NONE;
  }
};

SanWriter::SanWriter(const std::string& fen) : impl_(std::make_unique<Impl>()) {
  InitStockfish();
  impl_->states.emplace_back();
  impl_->pos.set(fen, false, &impl_->states.back(), &ThisThread());
}

SanWriter::~SanWriter() = default;

std::string SanWriter::San(const TacticsMove& move) const {
  Position& pos = impl_->pos;
  const Move m = impl_->Find(move);
  if (m == MOVE_NONE) return "";
  const Square from = from_sq(m);
  const Square to = to_sq(m);
  std::string san;
  if (type_of(m) == CASTLING) {
    san = to > from ? "O-O" : "O-O-O";
  } else {
    const PieceType pt = type_of(pos.moved_piece(m));
    const bool capture = pos.capture(m);
    if (pt == PAWN) {
      if (capture) san += char('a' + file_of(from));
    } else {
      san += " PNBRQK"[pt];
      // Disambiguation, by file if that is enough, then by rank, then both.
      bool other = false, same_file = false, same_rank = false;
      for (const ExtMove& e : MoveList<LEGAL>(pos)) {
        const Square f = from_sq(e.move);
        if (e.move == m || to_sq(e.move) != to || f == from ||
            type_of(pos.moved_piece(e.move)) != pt ||
            type_of(e.move) == CASTLING) {
          continue;
        }
        other = true;
        same_file |= file_of(f) == file_of(from);
        same_rank |= rank_of(f) == rank_of(from);
      }
      if (other) {
        if (!same_file) {
          san += char('a' + file_of(from));
        } else if (!same_rank) {
          san += char('1' + rank_of(from));
        } else {
          san += char('a' + file_of(from));
          san += char('1' + rank_of(from));
        }
      }
    }
    if (capture) san += 'x';
    san += char('a' + file_of(to));
    san += char('1' + rank_of(to));
    if (type_of(m) == PROMOTION) {
      san += '=';
      san += " PNBRQK"[promotion_type(m)];
    }
  }
  StateInfo st;
  pos.do_move(m, st);
  if (pos.checkers()) san += MoveList<LEGAL>(pos).size() ? '+' : '#';
  pos.undo_move(m);
  return san;
}

void SanWriter::Push(const TacticsMove& move) {
  const Move m = impl_->Find(move);
  impl_->states.emplace_back();
  impl_->pos.do_move(m, impl_->states.back());
  impl_->played.push_back(m);
}

void SanWriter::Pop() {
  impl_->pos.undo_move(impl_->played.back());
  impl_->played.pop_back();
  impl_->states.pop_back();
}

}  // namespace classic
}  // namespace lczero
