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

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#pragma once

#include <memory>

#include "neural/backend.h"
#include "utils/optionsparser.h"

namespace lczero {

// Opponent modeling, for testing which of a fine-tuned network's heads
// matter: the policy (PolicyWeights) or the moves left (MovesLeftWeights)
// comes from another network, each network evaluating with its own body,
// and the rest from the main one.
extern const OptionId kPolicyWeightsId;
extern const OptionId kMovesLeftWeightsId;

void PopulateSplitHeadsOptions(OptionsParser* parser);

// The main backend as it is without PolicyWeights and MovesLeftWeights;
// with either, a backend that takes those heads from their networks
// instead, created with the same backend settings.
std::unique_ptr<Backend> MaybeSplitHeads(std::unique_ptr<Backend> main,
                                         const OptionsDict& options);

}  // namespace lczero
