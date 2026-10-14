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

#include "search/classic/opponent_model.h"

#include <algorithm>
#include <cmath>

#include "neural/memcache.h"
#ifdef USE_MAIA3
#include "neural/backends/network_maia3.h"
#endif
#include "neural/register.h"
#include "neural/shared_params.h"
#include "utils/exception.h"

namespace lczero {
namespace classic {

const OptionId kOpponentModelWeightsId{
    {.long_flag = "opponent-model-weights",
     .uci_option = "OpponentModelWeights",
     .help_text =
         "Network that predicts the opponent's moves, such as a Maia network. "
         "When set, the opponent's moves are sampled from it in proportion to "
         "its move probabilities, instead of searched for the best. A .onnx "
         "file is read as a Maia-3 network (see tools/export_maia3.py). Empty "
         "turns opponent modeling off.",
     .visibility = OptionId::kAlwaysVisible}};
const OptionId kOpponentModelTemperatureId{
    {.long_flag = "opponent-model-temperature",
     .uci_option = "OpponentModelTemperature",
     .help_text =
         "Temperature for the opponent model's move probabilities. Above 1 "
         "makes the opponent less predictable, below 1 more.",
     .visibility = OptionId::kAlwaysVisible}};
const OptionId kOpponentModelRatingId{
    {.long_flag = "opponent-model-rating",
     .uci_option = "OpponentModelRating",
     .help_text = "Rating of the modeled opponent, for Maia-3 opponent models.",
     .visibility = OptionId::kAlwaysVisible}};
const OptionId kOpponentModelOwnRatingId{
    {.long_flag = "opponent-model-own-rating",
     .uci_option = "OpponentModelOwnRating",
     .help_text =
         "Rating the Maia-3 opponent model is told the opponent is playing "
         "against, that is the rating of the side this engine plays.",
     .visibility = OptionId::kAlwaysVisible}};

void PopulateOpponentModelOptions(OptionsParser* parser) {
  parser->Add<StringOption>(kOpponentModelWeightsId) = "";
  parser->Add<FloatOption>(kOpponentModelTemperatureId, 0.1f, 10.0f) = 1.0f;
  parser->Add<FloatOption>(kOpponentModelRatingId, 500.0f, 3000.0f) = 1500.0f;
  parser->Add<FloatOption>(kOpponentModelOwnRatingId, 500.0f, 3000.0f) =
      1500.0f;
}

std::unique_ptr<Backend> CreateOpponentModelBackend(
    const OptionsDict& options, std::unique_ptr<OptionsDict>* backend_options) {
  const std::string weights = options.Get<std::string>(kOpponentModelWeightsId);
  if (weights.empty()) return nullptr;
  const float temperature = options.Get<float>(kOpponentModelTemperatureId);

  // The same backend settings as the main network, with the opponent model's
  // weights and temperature. Lc0 applies the temperature inside the backend.
  *backend_options = std::make_unique<OptionsDict>(&options);
  (*backend_options)->Set<std::string>(SharedBackendParams::kWeightsId, weights);
  (*backend_options)
      ->Set<float>(SharedBackendParams::kPolicySoftmaxTemp, temperature);
  if (weights.ends_with(".onnx")) {
#ifdef USE_MAIA3
    // Maia-3 is not an Lc0 network and has its own backend
    return CreateMemCache(
        MakeMaia3Backend(
            {.model_path = weights,
             .temperature = temperature,
             .rating = options.Get<float>(kOpponentModelRatingId),
             .opponent_rating = options.Get<float>(kOpponentModelOwnRatingId)}),
        **backend_options);
#else
    throw Exception("This Lc0 was built without ONNX Runtime, so it cannot "
                    "run Maia-3 opponent models.");
#endif
  }
  return CreateMemCache(
      BackendManager::Get()->CreateFromParams(**backend_options),
      **backend_options);
}

void MixInEnginePolicy(std::span<float> policy,
                       std::span<const float> engine_policy, float mix) {
  for (size_t i = 0; i < policy.size(); ++i) {
    policy[i] += mix * (engine_policy[i] - policy[i]);
  }
}

void ShiftForQuickTactics(std::span<float> policy, std::span<const int> scores,
                          float max_rate) {
  constexpr float kLossScaleCp = 300.0f;
  size_t best = 0;
  for (size_t i = 1; i < policy.size(); ++i) {
    if (scores[i] > scores[best] ||
        (scores[i] == scores[best] && policy[i] > policy[best])) {
      best = i;
    }
  }
  float caught = 0.0f;
  for (size_t i = 0; i < policy.size(); ++i) {
    const float loss = static_cast<float>(scores[best] - scores[i]);
    const float rate = max_rate * (1.0f - std::exp(-loss / kLossScaleCp));
    caught += policy[i] * rate;
    policy[i] *= 1.0f - rate;
  }
  policy[best] += caught;
}

void DropUnlikelyMoves(std::span<float> policy, float min_probability) {
  if (policy.empty()) return;
  const float top = *std::max_element(policy.begin(), policy.end());
  float kept = 0.0f;
  for (float& p : policy) {
    if (p < min_probability && p < top) p = 0.0f;
    kept += p;
  }
  for (float& p : policy) p /= kept;
}

void ShiftForCloserLook(std::span<float> policy, std::span<const int> scores,
                        float strength) {
  constexpr float kLossScaleCp = 300.0f;
  const int best_score = *std::max_element(scores.begin(), scores.end());
  float total = 0.0f;
  for (size_t i = 0; i < policy.size(); ++i) {
    const float loss = static_cast<float>(best_score - scores[i]);
    policy[i] *= std::exp(-strength * loss / kLossScaleCp);
    total += policy[i];
  }
  if (total > 0.0f) {
    for (float& p : policy) p /= total;
  }
}

}  // namespace classic
}  // namespace lczero
