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

#include "chess/gamestate.h"
#include "search/classic/search.h"
#include "search/classic/stoppers/factory.h"
#include "neural/memcache.h"
#ifdef USE_MAIA3
#include "neural/backends/network_maia3.h"
#endif
#include "neural/register.h"
#include "search/register.h"
#include "search/search.h"
#include "src/neural/shared_params.h"

namespace lczero {
namespace classic {
namespace {

const OptionId kThreadsOptionId{
    {.long_flag = "threads",
     .uci_option = "Threads",
     .help_text =
         "Number of (CPU) worker threads to use, 0 for the backend default.",
     .short_flag = 't',
     .visibility = OptionId::kAlwaysVisible}};
const OptionId kClearTree{
    {.long_flag = "",
     .uci_option = "ClearTree",
     .help_text = "Clear the tree before the next search.",
     .visibility = OptionId::kProOnly}};

// Opponent modeling (see LC0_PLAN.md)
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
const OptionId kOpponentTreeReuseId{
    {.long_flag = "opponent-tree-reuse",
     .uci_option = "OpponentTreeReuse",
     .help_text =
         "With an opponent model, whether a search keeps the previous search's "
         "tree and network caches. The root compares its candidate moves at "
         "equal visits (OpponentRootCandidates), and visits kept from the "
         "previous search are uneven, which favors the moves it searched most. "
         "The caches hold evaluations of positions reached by other move "
         "orders, which differ a little from fresh ones, enough to change "
         "close decisions.",
     .visibility = OptionId::kAlwaysVisible}};

class ClassicSearch : public SearchBase {
 public:
  ClassicSearch(UciResponder* responder, const OptionsDict* options)
      : SearchBase(responder), options_(options) {}

 private:
  void NewGame() override;
  void SetPosition(const GameState& pos) override;
  void StartSearch(const GoParams&) override;
  void StartClock() override {
    move_start_time_ = std::chrono::steady_clock::now();
  }
  void WaitSearch() override {
    if (search_) search_->Wait();
  }
  void StopSearch() override {
    if (search_) search_->Stop();
  }
  void AbortSearch() override {
    if (search_) search_->Abort();
  }

  // Opponent modeling: creates or removes the opponent model's backend when its
  // options change, and clears the tree when its values no longer apply.
  void UpdateOpponentModel();

  const OptionsDict* options_;
  std::unique_ptr<TimeManager> time_manager_;
  std::unique_ptr<Search> search_;
  std::unique_ptr<NodeTree> tree_;

  // Opponent modeling: the backend running the opponent model, the options it
  // was created with, and what the tree was last searched with.
  std::unique_ptr<OptionsDict> opponent_options_;
  std::unique_ptr<Backend> opponent_backend_;
  std::string opponent_weights_;
  float opponent_temperature_ = 1.0f;
  float opponent_rating_ = 0.0f;
  float opponent_own_rating_ = 0.0f;
  bool opponent_black_to_move_ = false;
  std::optional<std::chrono::steady_clock::time_point> move_start_time_;
};

MoveList StringsToMovelist(const std::vector<std::string>& moves,
                           const ChessBoard& board) {
  MoveList result;
  if (moves.size()) {
    result.reserve(moves.size());
    const auto legal_moves = board.GenerateLegalMoves();
    for (const auto& move : moves) {
      const Move m = board.ParseMove(move);
      if (std::find(legal_moves.begin(), legal_moves.end(), m) !=
          legal_moves.end()) {
        result.emplace_back(m);
      }
    }
    if (result.empty()) throw Exception("No legal searchmoves.");
  }
  return result;
}

void ClassicSearch::NewGame() {
  search_.reset();
  tree_.reset();
  time_manager_ = MakeTimeManager(*options_);
}

void ClassicSearch::SetPosition(const GameState& pos) {
  if (!tree_) tree_ = std::make_unique<NodeTree>();
  const bool is_same_game = tree_->ResetToPosition(pos);
  if (!is_same_game) time_manager_ = MakeTimeManager(*options_);
}

void ClassicSearch::UpdateOpponentModel() {
  const std::string weights =
      options_->Get<std::string>(kOpponentModelWeightsId);
  const float temperature = options_->Get<float>(kOpponentModelTemperatureId);
  const float rating = options_->Get<float>(kOpponentModelRatingId);
  const float own_rating = options_->Get<float>(kOpponentModelOwnRatingId);
  const bool is_maia3 = weights.ends_with(".onnx");
  const bool black_to_move = tree_->HeadPosition().IsBlackToMove();
  const bool model_changed =
      weights != opponent_weights_ ||
      (!weights.empty() && temperature != opponent_temperature_) ||
      (is_maia3 &&
       (rating != opponent_rating_ || own_rating != opponent_own_rating_));

  // The tree's values and opponent-node move probabilities depend on the
  // model and on which side is the opponent: the side not to move at the root.
  // Unless reuse is asked for, every search starts afresh, since the root
  // compares moves at equal visits and kept visits are uneven.
  if (model_changed ||
      (!weights.empty() && (black_to_move != opponent_black_to_move_ ||
                            !options_->Get<bool>(kOpponentTreeReuseId)))) {
    tree_->TrimTreeAtHead();
  }
  opponent_black_to_move_ = black_to_move;

  if (!model_changed) return;
  opponent_backend_.reset();
  opponent_options_.reset();
  // Recorded only once the model has loaded, so a model that failed to load
  // is tried again rather than taken as loaded
  opponent_weights_.clear();
  if (weights.empty()) return;

  // The same backend settings as the main network, with the opponent model's
  // weights and temperature. Lc0 applies the temperature inside the backend.
  opponent_options_ = std::make_unique<OptionsDict>(options_);
  opponent_options_->Set<std::string>(SharedBackendParams::kWeightsId, weights);
  opponent_options_->Set<float>(SharedBackendParams::kPolicySoftmaxTemp,
                                temperature);
  if (is_maia3) {
#ifdef USE_MAIA3
    // Maia-3 is not an Lc0 network and has its own backend
    opponent_backend_ = CreateMemCache(
        MakeMaia3Backend({.model_path = weights,
                          .temperature = temperature,
                          .rating = rating,
                          .opponent_rating = own_rating}),
        *opponent_options_);
#else
    throw Exception("This Lc0 was built without ONNX Runtime, so it cannot "
                    "run Maia-3 opponent models.");
#endif
  } else {
    opponent_backend_ = CreateMemCache(
        BackendManager::Get()->CreateFromParams(*opponent_options_),
        *opponent_options_);
  }
  opponent_weights_ = weights;
  opponent_temperature_ = temperature;
  opponent_rating_ = rating;
  opponent_own_rating_ = own_rating;
}

void ClassicSearch::StartSearch(const GoParams& params) {
  auto forwarder =
      std::make_unique<NonOwningUciRespondForwarder>(uci_responder_);
  if (options_->Get<Button>(kClearTree).TestAndReset()) tree_->TrimTreeAtHead();
  UpdateOpponentModel();
  // Opponent modeling: unless reuse is asked for, every search also starts
  // with empty caches, so that it gives the same result as in a new process.
  // The caches are keyed by the position alone, while the networks also read
  // the moves before it, so an evaluation the previous search cached for the
  // same position, reached by another move order, differs a little from a
  // fresh one. That shifted values by about 0.01, enough to change the move.
  if (opponent_backend_ && !options_->Get<bool>(kOpponentTreeReuseId)) {
    for (Backend* backend : {backend_, opponent_backend_.get()}) {
      if (auto* cache = dynamic_cast<CachingBackend*>(backend)) {
        cache->ClearCache();
      }
    }
  }

  const auto cache_size =
      options_->Get<int>(SharedBackendParams::kNNCacheSizeId);
  const size_t kAvgNodeSize =
      sizeof(Node) + MemoryWatchingStopper::kAvgMovesPerPosition * sizeof(Edge);
  const size_t kAvgCacheItemSize =
      3 * sizeof(float) + sizeof(std::unique_ptr<float[]>) +
      sizeof(float[MemoryWatchingStopper::kAvgMovesPerPosition]);
  size_t total_memory = tree_.get()->GetCurrentHead()->GetN() * kAvgNodeSize +
                        cache_size * kAvgCacheItemSize;
  auto stopper = time_manager_->GetStopper(
      params, tree_.get()->HeadPosition(), total_memory, kAvgNodeSize,
      tree_.get()->GetCurrentHead()->GetN());
  search_ = std::make_unique<Search>(
      *tree_, backend_, std::move(forwarder),
      StringsToMovelist(params.searchmoves, tree_->HeadPosition().GetBoard()),
      *move_start_time_, std::move(stopper), params.infinite, params.ponder,
      *options_, syzygy_tb_, opponent_backend_.get());

  LOGFILE << "Timer started at "
          << FormatTime(SteadyClockToSystemClock(*move_start_time_));
  search_->StartThreads(options_->Get<int>(kThreadsOptionId));
}

class ClassicSearchFactory : public SearchFactory {
  std::string_view GetName() const override { return "classic"; }
  std::unique_ptr<SearchBase> CreateSearch(
      UciResponder* responder, const OptionsDict* options) const override {
    return std::make_unique<ClassicSearch>(responder, options);
  }

  void PopulateParams(OptionsParser* parser) const override {
    parser->Add<IntOption>(kThreadsOptionId, 0, 128) = 0;
    SearchParams::Populate(parser);
    PopulateTimeManagementOptions(RunType::kUci, parser);

    parser->Add<ButtonOption>(kClearTree);
    parser->Add<StringOption>(kOpponentModelWeightsId) = "";
    parser->Add<FloatOption>(kOpponentModelTemperatureId, 0.1f, 10.0f) = 1.0f;
    parser->Add<FloatOption>(kOpponentModelRatingId, 500.0f, 3000.0f) = 1500.0f;
    parser->Add<FloatOption>(kOpponentModelOwnRatingId, 500.0f, 3000.0f) =
        1500.0f;
    parser->Add<BoolOption>(kOpponentTreeReuseId) = false;
  }
};

REGISTER_SEARCH(ClassicSearchFactory);

}  // namespace
}  // namespace classic
}  // namespace lczero
