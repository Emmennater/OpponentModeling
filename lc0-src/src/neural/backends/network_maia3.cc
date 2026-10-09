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

#include "neural/backends/network_maia3.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <filesystem>
#include <future>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include "chess/position.h"
#include "onnxruntime_cxx_api.h"
#include "utils/exception.h"
#include "utils/logging.h"

namespace lczero {
namespace {

// The exported model's interface (see tools/export_maia3.py).
constexpr int kHistory = 8;                 // Positions in the input
constexpr int kPlanes = 12;                 // Piece planes per position
constexpr int kWidth = kHistory * kPlanes;  // Inputs per square
constexpr int kVocab = 4352;                // Move logits per position

// Batch sizes the network is prepared for. A batch runs at the smallest that
// holds it, padded with leftover positions.
constexpr std::array<int, 5> kBatchSizes = {16, 32, 64, 128, 256};
// Runs made when the network is loaded, by which time ONNX Runtime has
// recorded its CUDA graph (after one or two ordinary runs) and replays it.
constexpr int kWarmupRuns = 4;

// Index of a move in Maia-3's vocabulary: from * 64 + to, then promotions from
// rank 7 to rank 8 at 4096 + (from file * 8 + to file) * 4 + piece, pieces
// ordered q, r, b, n. Squares are numbered a1 = 0 ... h8 = 63 in the side to
// move's frame, which is the frame Lc0's moves are already in.
int VocabIndex(Move move) {
  const int from = move.from().as_idx();
  int to = move.to().as_idx();
  if (move.is_castling()) {
    // Lc0 writes castling as the king taking its own rook. Maia-3 writes the
    // king's two-square move.
    to = (from / 8) * 8 + (to % 8 > from % 8 ? 6 : 2);
  }
  if (move.is_promotion()) {
    const PieceType piece = move.promotion();
    const int piece_idx = piece == kQueen    ? 0
                          : piece == kRook   ? 1
                          : piece == kBishop ? 2
                                             : 3;
    return 4096 + ((from % 8) * 8 + (to % 8)) * 4 + piece_idx;
  }
  return from * 64 + to;
}

// Writes one position's 12 piece planes into slot `slot` of each square's
// inputs: pawn, knight, bishop, rook, queen, king of the side to move, then
// the same for its opponent. Lc0's board is already in the side to move's
// frame (ranks flipped for black), which is the frame Maia-3 expects.
void EncodeBoard(const ChessBoard& board, int slot, float* tokens) {
  const std::array<BitBoard, 6> pieces = {board.pawns(),   board.knights(),
                                          board.bishops(), board.rooks(),
                                          board.queens(),  board.kings()};
  for (int type = 0; type < 6; ++type) {
    for (Square sq : pieces[type] & board.ours()) {
      tokens[sq.as_idx() * kWidth + slot * kPlanes + type] = 1.0f;
    }
    for (Square sq : pieces[type] & board.theirs()) {
      tokens[sq.as_idx() * kWidth + slot * kPlanes + 6 + type] = 1.0f;
    }
  }
}

// Encodes the last kHistory positions, oldest first, each in its own frame.
// A shorter history is padded at the front with its oldest position, as the
// reference encoder (StockfishTrapFinder's Maia3Net) does.
void EncodeHistory(std::span<const Position> history, float* tokens) {
  std::fill(tokens, tokens + 64 * kWidth, 0.0f);
  const int available = std::min<int>(history.size(), kHistory);
  const size_t first = history.size() - available;
  for (int slot = 0; slot < kHistory; ++slot) {
    const int back = std::max(0, slot - (kHistory - available));
    EncodeBoard(history[first + back].GetBoard(), slot, tokens);
  }
}

Ort::Env& OrtEnvironment() {
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "lc0-maia3");
  return env;
}

class Maia3Backend;

class Maia3Computation : public BackendComputation {
 public:
  explicit Maia3Computation(Maia3Backend* backend) : backend_(backend) {}

  size_t UsedBatchSize() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
  }

  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    Entry entry{.result = result};
    entry.indices.reserve(pos.legal_moves.size());
    for (Move move : pos.legal_moves) entry.indices.push_back(VocabIndex(move));

    std::array<float, 64 * kWidth> tokens;
    EncodeHistory(pos.pos, tokens.data());

    // Search task workers may add inputs to the same computation in parallel
    std::lock_guard<std::mutex> lock(mutex_);
    tokens_.insert(tokens_.end(), tokens.begin(), tokens.end());
    entries_.push_back(std::move(entry));
    return ENQUEUED_FOR_EVAL;
  }

  void ComputeBlocking() override;

 private:
  struct Entry {
    EvalResultPtr result;
    std::vector<int> indices;  // Vocabulary index of each legal move
  };

  Maia3Backend* const backend_;
  mutable std::mutex mutex_;
  std::vector<float> tokens_;
  std::vector<Entry> entries_;
};

void CheckCuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw Exception(std::string("Maia-3 CUDA error: ") +
                    cudaGetErrorString(status));
  }
}

// The network prepared for one batch size. Running Maia-3 is mostly launching
// many small GPU operations, which costs about 12 ms a run whatever the batch
// size. So each batch size gets its own session with fixed shapes, which lets
// ONNX Runtime record the run once as a CUDA graph and replay it, and fixed
// GPU buffers for the inputs and outputs, which CUDA graphs need.
//
// The copies in and out and the run all go on one CUDA stream of our own, so
// CUDA runs them in order. ONNX Runtime otherwise runs on a stream of its own
// that does not wait for plain cudaMemcpy calls, and a run could read a batch
// before it had finished arriving, or its results be read before they were
// written: about half of all batches came out wrong under a busy search.
class Maia3Runner {
 public:
  Maia3Runner(const Maia3Options& options, int batch_size)
      : batch_size_(batch_size),
        stream_(CreateStream(options.gpu)),
        session_(OrtEnvironment(), ModelPath(options.model_path).c_str(),
                 SessionOptions(options.gpu, batch_size, stream_)) {
    CheckCuda(cudaMalloc(&tokens_, Bytes(64 * kWidth)));
    CheckCuda(cudaMalloc(&self_elo_, Bytes(1)));
    CheckCuda(cudaMalloc(&oppo_elo_, Bytes(1)));
    CheckCuda(cudaMalloc(&policy_, Bytes(kVocab)));
    CheckCuda(cudaMalloc(&wdl_, Bytes(3)));
    CheckCuda(cudaMemset(tokens_, 0, Bytes(64 * kWidth)));

    // The ratings are the same for every position
    const std::vector<float> self_elo(batch_size, options.rating);
    const std::vector<float> oppo_elo(batch_size, options.opponent_rating);
    CheckCuda(cudaMemcpy(self_elo_, self_elo.data(), Bytes(1),
                         cudaMemcpyHostToDevice));
    CheckCuda(cudaMemcpy(oppo_elo_, oppo_elo.data(), Bytes(1),
                         cudaMemcpyHostToDevice));

    const Ort::MemoryInfo memory("Cuda", OrtDeviceAllocator, options.gpu,
                                 OrtMemTypeDefault);
    const int64_t n = batch_size;
    const std::array<int64_t, 3> tokens_shape = {n, 64, kWidth};
    const std::array<int64_t, 1> elo_shape = {n};
    const std::array<int64_t, 2> policy_shape = {n, kVocab};
    const std::array<int64_t, 2> wdl_shape = {n, 3};
    auto tensor = [&](float* data, size_t per_position, const auto& shape) {
      return Ort::Value::CreateTensor<float>(memory, data, n * per_position,
                                             shape.data(), shape.size());
    };
    values_.push_back(tensor(tokens_, 64 * kWidth, tokens_shape));
    values_.push_back(tensor(self_elo_, 1, elo_shape));
    values_.push_back(tensor(oppo_elo_, 1, elo_shape));
    values_.push_back(tensor(policy_, kVocab, policy_shape));
    values_.push_back(tensor(wdl_, 3, wdl_shape));
    binding_.BindInput("tokens", values_[0]);
    binding_.BindInput("self_elo", values_[1]);
    binding_.BindInput("oppo_elo", values_[2]);
    binding_.BindOutput("policy", values_[3]);
    binding_.BindOutput("wdl", values_[4]);

    // ONNX Runtime records the CUDA graph during the first runs. Recording
    // forbids CUDA calls from other threads, such as the main network's
    // backend, so it is done now, before the search starts.
    for (int i = 0; i < kWarmupRuns; ++i) {
      session_.Run(Ort::RunOptions{nullptr}, binding_);
    }
    CheckCuda(cudaDeviceSynchronize());
  }

  ~Maia3Runner() {
    cudaStreamDestroy(stream_);
    cudaFree(tokens_);
    cudaFree(self_elo_);
    cudaFree(oppo_elo_);
    cudaFree(policy_);
    cudaFree(wdl_);
  }

  int batch_size() const { return batch_size_; }

  // Runs `count` positions, at most batch_size(): tokens holds 64 * kWidth
  // inputs for each, and policy and wdl receive kVocab and 3 logits for each.
  // The rest of the batch reruns whatever positions were last there.
  void Run(const float* tokens, int count, float* policy, float* wdl) {
    CheckCuda(cudaMemcpyAsync(tokens_, tokens,
                              count * 64 * kWidth * sizeof(float),
                              cudaMemcpyHostToDevice, stream_));
    session_.Run(Ort::RunOptions{nullptr}, binding_);
    CheckCuda(cudaMemcpyAsync(policy, policy_, count * kVocab * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_));
    CheckCuda(cudaMemcpyAsync(wdl, wdl_, count * 3 * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_));
    CheckCuda(cudaStreamSynchronize(stream_));
  }

 private:
  size_t Bytes(size_t per_position) const {
    return batch_size_ * per_position * sizeof(float);
  }

  static std::filesystem::path ModelPath(const std::string& path) {
    if (!std::filesystem::exists(path)) {
      throw Exception("Opponent model file not found: " + path);
    }
    return std::filesystem::path(path);
  }

  static cudaStream_t CreateStream(int gpu) {
    CheckCuda(cudaSetDevice(gpu));
    cudaStream_t stream;
    CheckCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    return stream;
  }

  static Ort::SessionOptions SessionOptions(int gpu, int batch_size,
                                            cudaStream_t stream) {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // A fixed batch size lets the shape computations be folded away, which
    // leaves only GPU operations, as CUDA graphs require.
    options.AddFreeDimensionOverrideByName("batch", batch_size);
    Ort::CUDAProviderOptions cuda_options;
    cuda_options.Update({{"device_id", std::to_string(gpu)},
                         {"enable_cuda_graph", "1"}});
    cuda_options.UpdateWithValue("user_compute_stream", stream);
    options.AppendExecutionProvider_CUDA_V2(*cuda_options);
    return options;
  }

  const int batch_size_;
  const cudaStream_t stream_;
  Ort::Session session_;
  Ort::IoBinding binding_{session_};
  std::vector<Ort::Value> values_;
  float* tokens_ = nullptr;
  float* self_elo_ = nullptr;
  float* oppo_elo_ = nullptr;
  float* policy_ = nullptr;
  float* wdl_ = nullptr;
};

// Runs Maia-3 on one thread of its own. ONNX Runtime keeps a session's CUDA
// graph per calling thread, and Lc0 starts new search threads for every
// search, so with the search threads calling it, every search recorded the
// graphs again. Recording blocks CUDA calls from other threads, and when the
// main network made one meanwhile, Lc0 failed with "operation not permitted
// when stream is capturing". Here the graphs are recorded once, when the
// worker thread sets the sessions up, and only replayed after that.
class Maia3Backend : public Backend {
 public:
  explicit Maia3Backend(const Maia3Options& options) : options_(options) {
    std::promise<void> ready;
    std::future<void> started = ready.get_future();
    worker_ = std::thread([this, &ready] { Work(ready); });
    try {
      started.get();
    } catch (...) {
      worker_.join();
      throw;
    }
    CERR << "Opponent model: Maia-3 from " << options.model_path
         << ", rating " << options.rating << " against "
         << options.opponent_rating;
  }

  ~Maia3Backend() override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      quit_ = true;
    }
    job_ready_.notify_all();
    worker_.join();
  }

  BackendAttributes GetAttributes() const override {
    return BackendAttributes{
        .has_mlh = false,
        .has_wdl = true,
        .runs_on_cpu = false,
        .suggested_num_search_threads = 0,
        .recommended_batch_size = max_batch(),
        .maximum_batch_size = 1024,
    };
  }

  std::unique_ptr<BackendComputation> CreateComputation() override {
    return std::make_unique<Maia3Computation>(this);
  }

  // The largest batch run at once
  int max_batch() const { return max_batch_; }

  // Runs `count` positions, at most max_batch(), as Maia3Runner::Run does, on
  // the worker thread, and waits for the result.
  void Run(const float* tokens, int count, float* policy, float* wdl) {
    // One batch at a time: the runners' GPU buffers are shared
    std::lock_guard<std::mutex> run_lock(run_mutex_);
    std::unique_lock<std::mutex> lock(mutex_);
    job_ = Job{tokens, count, policy, wdl};
    job_done_ = false;
    job_ready_.notify_all();
    job_finished_.wait(lock, [this] { return job_done_; });
    if (job_error_) std::rethrow_exception(std::exchange(job_error_, nullptr));
  }

  const Maia3Options& options() const { return options_; }

 private:
  struct Job {
    const float* tokens;
    int count;
    float* policy;
    float* wdl;
  };

  void Work(std::promise<void>& ready) {
    std::vector<std::unique_ptr<Maia3Runner>> runners;
    try {
      CheckCuda(cudaSetDevice(options_.gpu));
      for (int batch_size : kBatchSizes) {
        if (batch_size > options_.max_batch && !runners.empty()) break;
        runners.push_back(std::make_unique<Maia3Runner>(options_, batch_size));
      }
      max_batch_ = runners.back()->batch_size();
    } catch (...) {
      ready.set_exception(std::current_exception());
      return;
    }
    ready.set_value();

    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
      job_ready_.wait(lock, [this] { return quit_ || job_.has_value(); });
      if (quit_) break;
      const Job job = *job_;
      job_.reset();
      lock.unlock();
      std::exception_ptr error;
      try {
        const auto runner = std::find_if(
            runners.begin(), runners.end(),
            [&](const auto& r) { return r->batch_size() >= job.count; });
        if (runner == runners.end()) {
          throw Exception("Maia-3 batch larger than the largest batch size");
        }
        (*runner)->Run(job.tokens, job.count, job.policy, job.wdl);
      } catch (...) {
        error = std::current_exception();
      }
      lock.lock();
      job_error_ = error;
      job_done_ = true;
      job_finished_.notify_all();
    }
    // The sessions are released on the thread that used them
    lock.unlock();
    runners.clear();
  }

  const Maia3Options options_;
  int max_batch_ = 0;
  std::thread worker_;
  std::mutex run_mutex_;
  std::mutex mutex_;
  std::condition_variable job_ready_;
  std::condition_variable job_finished_;
  std::optional<Job> job_;
  bool job_done_ = false;
  bool quit_ = false;
  std::exception_ptr job_error_;
};

void Maia3Computation::ComputeBlocking() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (entries_.empty()) return;

  const int max_batch = backend_->max_batch();
  const float inverse_temperature = 1.0f / backend_->options().temperature;
  std::vector<float> policy;
  std::vector<float> wdl;

  for (size_t start = 0; start < entries_.size(); start += max_batch) {
    const int count =
        static_cast<int>(std::min<size_t>(max_batch, entries_.size() - start));
    policy.resize(static_cast<size_t>(count) * kVocab);
    wdl.resize(static_cast<size_t>(count) * 3);
    backend_->Run(tokens_.data() + start * 64 * kWidth, count, policy.data(),
                  wdl.data());

    for (int i = 0; i < count; ++i) {
      Entry& entry = entries_[start + i];
      const float* logits = policy.data() + static_cast<size_t>(i) * kVocab;

      // Softmax over the legal moves only, at the policy temperature
      float max_logit = -INFINITY;
      for (int idx : entry.indices) max_logit = std::max(max_logit, logits[idx]);
      float total = 0.0f;
      for (size_t j = 0; j < entry.indices.size(); ++j) {
        const float p =
            std::exp((logits[entry.indices[j]] - max_logit) * inverse_temperature);
        entry.result.p[j] = p;
        total += p;
      }
      for (size_t j = 0; j < entry.indices.size(); ++j) entry.result.p[j] /= total;

      // Value: Maia-3 orders it loss, draw, win for the player to move
      const float* v = wdl.data() + static_cast<size_t>(i) * 3;
      const float max_v = std::max({v[0], v[1], v[2]});
      const float loss = std::exp(v[0] - max_v), draw = std::exp(v[1] - max_v),
                  win = std::exp(v[2] - max_v);
      const float sum = loss + draw + win;
      if (entry.result.q) *entry.result.q = (win - loss) / sum;
      if (entry.result.d) *entry.result.d = draw / sum;
      if (entry.result.m) *entry.result.m = 0.0f;
    }
  }

  entries_.clear();
  tokens_.clear();
}

}  // namespace

std::unique_ptr<Backend> MakeMaia3Backend(const Maia3Options& options) {
  return std::make_unique<Maia3Backend>(options);
}

}  // namespace lczero
