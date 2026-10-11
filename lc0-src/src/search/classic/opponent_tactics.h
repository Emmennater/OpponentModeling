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

#include <atomic>
#include <memory>
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

// Opponent modeling: like ScoreOpponentMoves, but each move is scored by an
// alpha-beta search `depth` half-moves deep after it (checks extend the
// search), then the capture search, so that it sees tactics a few moves deep
// such as quiet checks and trapped pieces. Scores far below the best move's
// are only bounds (a loss of at least 8 pawns). Returns an empty vector if
// `stop` becomes true. Thread-safe.
std::vector<int> ScoreOpponentMovesDeep(const std::string& fen,
                                        std::span<const TacticsMove> moves,
                                        int depth,
                                        const std::atomic<bool>* stop);

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
