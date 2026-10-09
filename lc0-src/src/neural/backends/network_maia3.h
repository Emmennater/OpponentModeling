/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2025 The LCZero Authors

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
#include <string>

#include "neural/backend.h"

namespace lczero {

// Opponent modeling: a Maia-3 move-prediction network, exported to ONNX by
// tools/export_maia3.py and run with ONNX Runtime on CUDA. Maia-3 is not an
// Lc0 network: it reads 64 square tokens of 12 piece planes for each of the
// last 8 positions, takes both players' ratings as input, and has its own
// move vocabulary. See LC0_PLAN.md.
struct Maia3Options {
  std::string model_path;
  // Policy temperature: above 1 flattens the move probabilities.
  float temperature = 1.0f;
  // Rating of the player to move, the modeled opponent.
  float rating = 1300.0f;
  // Rating of their opponent, this engine as the model sees it.
  float opponent_rating = 1300.0f;
  int gpu = 0;
  // Largest batch sent to the network in one run.
  int max_batch = 256;
};

std::unique_ptr<Backend> MakeMaia3Backend(const Maia3Options& options);

}  // namespace lczero
