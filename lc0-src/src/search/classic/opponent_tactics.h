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

#include <memory>
#include <utility>
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
// `fen` by what a player sees of its material consequences at a glance: the
// capture sequences that follow it (every move when in check) on material
// alone, where a side that stops capturing also loses what it has left
// hanging. A move that leaves two pieces hanging after a queen trade then
// shows as losing them, even if a deeper search would find a fork that wins
// some back. The scores are in centipawns for the side to move; a mate is
// -10000. Uses Stockfish 11's board (third_party/stockfish11). Thread-safe.
std::vector<int> ScoreOpponentMoves(const std::string& fen,
                                    std::span<const TacticsMove> moves);

// The same for every legal move, with its UCI notation (castling as e1g1),
// for tools (see `lc0 visibleloss`).
std::vector<std::pair<std::string, int>> ScoreVisible(const std::string& fen);

// Writes moves in standard algebraic notation (SAN), for exporting the search
// tree as PGN, with Stockfish 11's board. Moves are played with Push() and
// taken back with Pop().
class SanWriter {
 public:
  explicit SanWriter(const std::string& fen);
  ~SanWriter();
  // The move's SAN in the current position, or "" if it is not legal there.
  std::string San(const TacticsMove& move) const;
  void Push(const TacticsMove& move);
  void Pop();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace classic
}  // namespace lczero
