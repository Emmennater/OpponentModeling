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

#include "neural/split_heads.h"

#include "neural/register.h"
#include "neural/shared_params.h"
#include "utils/exception.h"

namespace lczero {

const OptionId kPolicyWeightsId{
    "policy-weights", "PolicyWeights",
    "Opponent modeling, for testing: a second network whose policy the "
    "search uses instead of the main network's. Each network evaluates every "
    "position with its own body. All get the identity plane (IdentityPlane) "
    "if it is on. Empty uses the main network's policy."};
const OptionId kMovesLeftWeightsId{
    "moves-left-weights", "MovesLeftWeights",
    "Opponent modeling, for testing: a second network whose moves left the "
    "search uses instead of the main network's. Each network evaluates every "
    "position with its own body. All get the identity plane (IdentityPlane) "
    "if it is on. Empty uses the main network's moves left."};

void PopulateSplitHeadsOptions(OptionsParser* parser) {
  parser->Add<StringOption>(kPolicyWeightsId) = "";
  parser->Add<StringOption>(kMovesLeftWeightsId) = "";
}

namespace {

class SplitHeadsBackend : public Backend {
 public:
  // `policy` and `moves_left` may be null: that head is the main network's.
  SplitHeadsBackend(std::unique_ptr<Backend> main,
                    std::unique_ptr<Backend> policy,
                    std::unique_ptr<Backend> moves_left,
                    const OptionsDict& options)
      : main_(std::move(main)),
        policy_(std::move(policy)),
        moves_left_(std::move(moves_left)),
        policy_weights_(options.Get<std::string>(kPolicyWeightsId)),
        moves_left_weights_(options.Get<std::string>(kMovesLeftWeightsId)) {}

  BackendAttributes GetAttributes() const override {
    BackendAttributes attrs = main_->GetAttributes();
    if (policy_) {
      attrs.maximum_batch_size = std::min(
          attrs.maximum_batch_size, policy_->GetAttributes().maximum_batch_size);
    }
    if (moves_left_) {
      const BackendAttributes ml = moves_left_->GetAttributes();
      attrs.has_mlh = ml.has_mlh;
      attrs.maximum_batch_size =
          std::min(attrs.maximum_batch_size, ml.maximum_batch_size);
    }
    return attrs;
  }

  std::unique_ptr<BackendComputation> CreateComputation() override;

  bool IsSameConfiguration(const OptionsDict& options) const override {
    return main_->IsSameConfiguration(options) &&
           options.Get<std::string>(kPolicyWeightsId) == policy_weights_ &&
           options.Get<std::string>(kMovesLeftWeightsId) == moves_left_weights_;
  }

 private:
  std::unique_ptr<Backend> main_;
  std::unique_ptr<Backend> policy_;
  std::unique_ptr<Backend> moves_left_;
  const std::string policy_weights_;
  const std::string moves_left_weights_;
};

class SplitHeadsComputation : public BackendComputation {
 public:
  SplitHeadsComputation(std::unique_ptr<BackendComputation> main,
                        std::unique_ptr<BackendComputation> policy,
                        std::unique_ptr<BackendComputation> moves_left)
      : main_(std::move(main)),
        policy_(std::move(policy)),
        moves_left_(std::move(moves_left)) {}

  size_t UsedBatchSize() const override { return main_->UsedBatchSize(); }

  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    // The main network fills the heads the others do not. The moves left's
    // network gets no legal moves, as it gives no policy.
    main_->AddInput(pos, EvalResultPtr{
                             .q = result.q,
                             .d = result.d,
                             .m = moves_left_ ? nullptr : result.m,
                             .p = policy_ ? std::span<float>{} : result.p});
    if (policy_) policy_->AddInput(pos, EvalResultPtr{.p = result.p});
    if (moves_left_) {
      moves_left_->AddInput(
          EvalPosition{.pos = pos.pos,
                       .legal_moves = {},
                       .engine_to_move = pos.engine_to_move},
          EvalResultPtr{.m = result.m});
    }
    return ENQUEUED_FOR_EVAL;
  }

  void ComputeBlocking() override {
    main_->ComputeBlocking();
    if (policy_) policy_->ComputeBlocking();
    if (moves_left_) moves_left_->ComputeBlocking();
  }

 private:
  std::unique_ptr<BackendComputation> main_;
  std::unique_ptr<BackendComputation> policy_;
  std::unique_ptr<BackendComputation> moves_left_;
};

std::unique_ptr<BackendComputation> SplitHeadsBackend::CreateComputation() {
  return std::make_unique<SplitHeadsComputation>(
      main_->CreateComputation(),
      policy_ ? policy_->CreateComputation() : nullptr,
      moves_left_ ? moves_left_->CreateComputation() : nullptr);
}

// The network at `weights` with the main network's backend settings, or
// null without weights.
std::unique_ptr<Backend> CreateHeadBackend(const std::string& weights,
                                           const OptionsDict& options) {
  if (weights.empty()) return nullptr;
  OptionsDict head_options(&options);
  head_options.Set<std::string>(SharedBackendParams::kWeightsId, weights);
  return BackendManager::Get()->CreateFromParams(head_options);
}

}  // namespace

std::unique_ptr<Backend> MaybeSplitHeads(std::unique_ptr<Backend> main,
                                         const OptionsDict& options) {
  auto policy =
      CreateHeadBackend(options.Get<std::string>(kPolicyWeightsId), options);
  auto moves_left =
      CreateHeadBackend(options.Get<std::string>(kMovesLeftWeightsId), options);
  if (!policy && !moves_left) return main;
  if (moves_left && !moves_left->GetAttributes().has_mlh) {
    throw Exception("MovesLeftWeights needs a network with a moves left head.");
  }
  return std::make_unique<SplitHeadsBackend>(
      std::move(main), std::move(policy), std::move(moves_left), options);
}

}  // namespace lczero
