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

// The visible-loss check: what a player sees of a move's material
// consequences at a glance. Material counts pawns 100, knights and bishops 300,
// rooks 500 and queens 900; the capture sequences are played out (every move
// when in check), and wherever a side stops capturing, what it has left
// hanging counts against it: the most the other side could win at once, by a
// simple exchange. An ordinary quiescence search would let the side to move
// "stand pat" with pieces en prise, and a deeper search would find the
// tactics that win the material back, which a player at a glance does not.
constexpr int kPieceCp[] = {0, 100, 300, 300, 500, 900, 0};
constexpr int kVisibleMaxPly = 8;

int MaterialCp(const Position& pos) {
  const Color us = pos.side_to_move();
  int cp = 0;
  for (PieceType pt : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN}) {
    cp += kPieceCp[pt] * (popcount(pos.pieces(us, pt)) - popcount(pos.pieces(~us, pt)));
  }
  return cp;
}

// The most the side not to move could win at once from the side to move: a
// piece's value, less its cheapest attacker's if the piece is defended.
int HangingCp(const Position& pos) {
  const Color us = pos.side_to_move();
  const Color them = ~us;
  int most = 0;
  Bitboard ours = pos.pieces(us) & ~pos.pieces(KING);
  while (ours) {
    const Square sq = pop_lsb(&ours);
    const Bitboard attackers = pos.attackers_to(sq) & pos.pieces(them);
    if (!attackers) continue;
    int cheapest = 100000;
    for (PieceType pt : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
      if (attackers & pos.pieces(them, pt)) {
        cheapest = pt == KING ? 100000 : kPieceCp[pt];
        break;
      }
    }
    const bool defended = pos.attackers_to(sq) & pos.pieces(us);
    const int gain = kPieceCp[type_of(pos.piece_on(sq))] - (defended ? cheapest : 0);
    most = std::max(most, gain);
  }
  return most;
}

int VisibleCp(Position& pos, int alpha, int beta, int ply) {
  const bool in_check = pos.checkers();
  int best;
  ExtMove moves[MAX_MOVES];
  ExtMove* end;
  if (!in_check) {
    best = MaterialCp(pos) - HangingCp(pos);
    if (best >= beta || ply >= kVisibleMaxPly) return best;
    alpha = std::max(alpha, best);
    end = generate<CAPTURES>(pos, moves);
  } else {
    best = -kMateCp;
    end = generate<EVASIONS>(pos, moves);
  }
  for (ExtMove* m = moves; m != end; ++m) m->value = CaptureOrder(pos, m->move);
  std::sort(moves, end, [](const ExtMove& a, const ExtMove& b) {
    return a.value > b.value;
  });
  bool any_legal = false;
  StateInfo st;
  for (ExtMove* m = moves; m != end; ++m) {
    if (!pos.legal(m->move)) continue;
    any_legal = true;
    // Captures that lose material by exchange are not what a glance looks at.
    if (!in_check && !pos.see_ge(m->move)) continue;
    pos.do_move(m->move, st);
    const int value = -VisibleCp(pos, -beta, -alpha, ply + 1);
    pos.undo_move(m->move);
    if (value > best) {
      best = value;
      if (value > alpha) alpha = value;
      if (value >= beta) return value;
    }
  }
  if (in_check && !any_legal) return -kMateCp;
  return best;
}

std::string UciMove(Move m) {
  const Square from = from_sq(m);
  Square to = to_sq(m);
  if (type_of(m) == CASTLING) {  // As e1g1, not the king taking its rook.
    to = make_square(to > from ? FILE_G : FILE_C, rank_of(from));
  }
  std::string uci = {char('a' + file_of(from)), char('1' + rank_of(from)),
                     char('a' + file_of(to)), char('1' + rank_of(to))};
  if (type_of(m) == PROMOTION) uci += " pnbrqk"[promotion_type(m)];
  return uci;
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
    scores[Key(m.move)] = -VisibleCp(pos, -kMateCp, kMateCp, 1);
    pos.undo_move(m.move);
  }

  std::vector<int> result;
  result.reserve(moves.size());
  for (const TacticsMove& m : moves) {
    const auto it = scores.find(Key(m.from, m.to, m.promotion));
    result.push_back(it == scores.end() ? -kMateCp : it->second);
  }
  return result;
}

std::vector<std::pair<std::string, int>> ScoreVisible(const std::string& fen) {
  InitStockfish();
  Position pos;
  StateInfo root_state;
  pos.set(fen, false, &root_state, &ThisThread());
  std::vector<std::pair<std::string, int>> result;
  StateInfo st;
  for (const ExtMove& m : MoveList<LEGAL>(pos)) {
    pos.do_move(m.move, st);
    result.emplace_back(UciMove(m.move), -VisibleCp(pos, -kMateCp, kMateCp, 1));
    pos.undo_move(m.move);
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
