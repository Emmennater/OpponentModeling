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
#include <optional>
#include <string>
#include <vector>

namespace lczero {
namespace classic {

// Opponent modeling: a UCI engine, such as Stockfish, run as a separate
// process, which scores every legal move of a position (see
// OpponentDeepTactics). One instance per thread.
class UciEngine {
 public:
  // Starts the engine at `path` with one thread and a small hash table.
  // Returns null if it cannot be started.
  static std::unique_ptr<UciEngine> Start(const std::string& path);
  ~UciEngine();

  // An engine at `path` from those kept between searches, or a new one;
  // Release() returns it. Starting Stockfish takes a moment, which a short
  // search cannot spare.
  static std::unique_ptr<UciEngine> Acquire(const std::string& path);
  static void Release(const std::string& path, std::unique_ptr<UciEngine> engine);

  // Scores `moves` (in UCI notation, castling as e1g1) of the position `fen`
  // by a search `depth` half-moves deep, stopping after `max_ms` at most.
  // Scores are in centipawns for the side to move, a mate is +-10000, and a
  // move the engine did not report gets -10000. Returns nothing if the engine
  // stopped working.
  std::optional<std::vector<int>> Score(const std::string& fen,
                                        const std::vector<std::string>& moves,
                                        int depth, int max_ms);

 private:
  struct Process;
  explicit UciEngine(std::unique_ptr<Process> process);
  bool Send(const std::string& line);
  bool ReadLine(std::string* line);
  bool WaitFor(const std::string& prefix);

  std::unique_ptr<Process> process_;
  std::string buffer_;
};

}  // namespace classic
}  // namespace lczero
