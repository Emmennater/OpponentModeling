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
#include <span>

#include "neural/backend.h"
#include "utils/optionsdict.h"
#include "utils/optionsparser.h"

namespace lczero {
namespace classic {

// Opponent modeling (see LC0_PLAN.md): the modeled opponent's move
// probabilities, shared by the search, at its opponent nodes, and by
// self-play, where an opponent player plays from them (see
// selfplay/game.cc).

// The opponent model's network and how it is run.
extern const OptionId kOpponentModelWeightsId;
extern const OptionId kOpponentModelTemperatureId;
extern const OptionId kOpponentModelRatingId;
extern const OptionId kOpponentModelOwnRatingId;

// Adds the options above.
void PopulateOpponentModelOptions(OptionsParser* parser);

// The backend for the opponent model that `options` names, or null when
// OpponentModelWeights is empty. It runs with the same backend settings as
// the main network; *backend_options receives the options it reads, which
// must outlive it.
std::unique_ptr<Backend> CreateOpponentModelBackend(
    const OptionsDict& options, std::unique_ptr<OptionsDict>* backend_options);

// The shifts below change `policy`, the opponent model's probabilities for
// the legal moves, in place. They are applied in this order, each only when
// its option is above 0.

// A share of the main network's probabilities for the same moves, which
// makes the modeled opponent stronger (OpponentEngineMix).
void MixInEnginePolicy(std::span<float> policy,
                       std::span<const float> engine_policy, float mix);

// The opponent sees some of the tactics a quick look shows: each move gives
// up to `max_rate` of its probability to the best-scoring move, more the
// more it loses, as in StockfishTrapFinder's MaiaModel. `scores` are
// ScoreOpponentMoves()'s, in centipawns. Measured on real games, the
// opponent then finds clear tactical shots more often, while quiet positions
// change little (OpponentTactics, see LC0_PLAN.md).
void ShiftForQuickTactics(std::span<float> policy, std::span<const int> scores,
                          float max_rate);

// Moves below `min_probability` are taken as never played, except the most
// likely; the rest are renormalized (OpponentMinProbability).
void DropUnlikelyMoves(std::span<float> policy, float min_probability);

// The closer look: each move's probability is weighted by
// exp(-strength * loss / 300 cp), its loss against the best-scoring move by
// Stockfish's `scores`, and the probabilities renormalized: a blunder is the
// more likely to be spotted the more it loses, and a move's odds against
// another fall with how much worse it is. A plain defense the opponent model
// already favors then becomes the clear choice, while a hidden only move it
// gives little stays unlikely (OpponentDeepTactics).
void ShiftForCloserLook(std::span<float> policy, std::span<const int> scores,
                        float strength);

}  // namespace classic
}  // namespace lczero
