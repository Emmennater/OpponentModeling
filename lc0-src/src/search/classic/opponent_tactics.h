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

#pragma once

#include <span>
#include <string>
#include <vector>

namespace lczero {
namespace classic {

// Opponent modeling: a move to score, on the absolute board (a1 = 0, h8 = 63).
// Castling is the king taking its own rook. promotion is 'q', 'r', 'b' or
// 'n', or 0.
struct TacticsMove {
  int from;
  int to;
  char promotion;
};

// Opponent modeling: scores each move of the side to move in the position
// `fen` by a depth-1 search: the move, then the best captures for both sides
// (all moves when in check), judged by Stockfish 11's classical evaluation of
// material, piece placement, mobility, king safety and pawns. The scores are
// in centipawns for the side to move; a mate is +-10000. It is a quick look
// at what each move wins or loses, as a club player might take, and is
// implemented with Stockfish 11 (third_party/stockfish11). Thread-safe.
std::vector<int> ScoreOpponentMoves(const std::string& fen,
                                    std::span<const TacticsMove> moves);

}  // namespace classic
}  // namespace lczero
