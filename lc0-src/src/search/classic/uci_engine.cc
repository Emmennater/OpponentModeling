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

#include "search/classic/uci_engine.h"

#include <sstream>
#include <unordered_map>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace lczero {
namespace classic {

constexpr int kMateCp = 10000;

#ifdef _WIN32
struct UciEngine::Process {
  HANDLE process = nullptr;
  HANDLE to_engine = nullptr;
  HANDLE from_engine = nullptr;

  ~Process() {
    if (to_engine) CloseHandle(to_engine);
    if (process) {
      if (WaitForSingleObject(process, 1000) != WAIT_OBJECT_0) {
        TerminateProcess(process, 1);
      }
      CloseHandle(process);
    }
    if (from_engine) CloseHandle(from_engine);
  }

  static std::unique_ptr<Process> Start(const std::string& path) {
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE out_read, out_write, in_read, in_write;
    if (!CreatePipe(&out_read, &out_write, &sa, 0)) return nullptr;
    if (!CreatePipe(&in_read, &in_write, &sa, 0)) {
      CloseHandle(out_read);
      CloseHandle(out_write);
      return nullptr;
    }
    // The engine inherits only its own ends of the pipes.
    SetHandleInformation(out_read, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(in_write, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = in_read;
    si.hStdOutput = out_write;
    si.hStdError = out_write;
    PROCESS_INFORMATION pi{};
    std::string command = "\"" + path + "\"";
    const BOOL ok =
        CreateProcessA(nullptr, command.data(), nullptr, nullptr, TRUE,
                       CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    CloseHandle(out_write);
    CloseHandle(in_read);
    if (!ok) {
      CloseHandle(out_read);
      CloseHandle(in_write);
      return nullptr;
    }
    CloseHandle(pi.hThread);
    auto process = std::make_unique<Process>();
    process->process = pi.hProcess;
    process->to_engine = in_write;
    process->from_engine = out_read;
    return process;
  }

  bool Write(const std::string& data) {
    DWORD written;
    return WriteFile(to_engine, data.data(), static_cast<DWORD>(data.size()),
                     &written, nullptr) &&
           written == data.size();
  }

  // Reads some output; false at the end of it.
  bool Read(std::string* out) {
    char buffer[4096];
    DWORD read;
    if (!ReadFile(from_engine, buffer, sizeof(buffer), &read, nullptr) ||
        read == 0) {
      return false;
    }
    out->append(buffer, read);
    return true;
  }
};
#else
struct UciEngine::Process {
  pid_t pid = -1;
  int to_engine = -1;
  int from_engine = -1;

  ~Process() {
    if (to_engine >= 0) close(to_engine);
    if (pid > 0) {
      int status;
      for (int i = 0; i < 100 && waitpid(pid, &status, WNOHANG) == 0; ++i) {
        usleep(10000);
      }
      if (waitpid(pid, &status, WNOHANG) == 0) {
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
      }
    }
    if (from_engine >= 0) close(from_engine);
  }

  static std::unique_ptr<Process> Start(const std::string& path) {
    int in[2], out[2];
    if (pipe(in) != 0) return nullptr;
    if (pipe(out) != 0) {
      close(in[0]);
      close(in[1]);
      return nullptr;
    }
    const pid_t pid = fork();
    if (pid < 0) return nullptr;
    if (pid == 0) {
      dup2(in[0], 0);
      dup2(out[1], 1);
      dup2(out[1], 2);
      close(in[1]);
      close(out[0]);
      execl(path.c_str(), path.c_str(), static_cast<char*>(nullptr));
      _exit(127);
    }
    close(in[0]);
    close(out[1]);
    auto process = std::make_unique<Process>();
    process->pid = pid;
    process->to_engine = in[1];
    process->from_engine = out[0];
    return process;
  }

  bool Write(const std::string& data) {
    return write(to_engine, data.data(), data.size()) ==
           static_cast<ssize_t>(data.size());
  }

  bool Read(std::string* out) {
    char buffer[4096];
    const ssize_t n = read(from_engine, buffer, sizeof(buffer));
    if (n <= 0) return false;
    out->append(buffer, n);
    return true;
  }
};
#endif

UciEngine::UciEngine(std::unique_ptr<Process> process)
    : process_(std::move(process)) {}

UciEngine::~UciEngine() {
  if (process_) Send("quit");
}

std::unique_ptr<UciEngine> UciEngine::Start(const std::string& path) {
  auto process = Process::Start(path);
  if (!process) return nullptr;
  std::unique_ptr<UciEngine> engine(new UciEngine(std::move(process)));
  if (!engine->Send("uci") || !engine->WaitFor("uciok") ||
      !engine->Send("setoption name Threads value 1") ||
      !engine->Send("setoption name Hash value 16") ||
      !engine->Send("isready") || !engine->WaitFor("readyok")) {
    return nullptr;
  }
  return engine;
}

bool UciEngine::Send(const std::string& line) {
  return process_->Write(line + "\n");
}

bool UciEngine::ReadLine(std::string* line) {
  while (true) {
    const size_t end = buffer_.find('\n');
    if (end != std::string::npos) {
      *line = buffer_.substr(0, end);
      if (!line->empty() && line->back() == '\r') line->pop_back();
      buffer_.erase(0, end + 1);
      return true;
    }
    if (!process_->Read(&buffer_)) return false;
  }
}

bool UciEngine::WaitFor(const std::string& prefix) {
  std::string line;
  while (ReadLine(&line)) {
    if (line.rfind(prefix, 0) == 0) return true;
  }
  return false;
}

std::optional<std::vector<int>> UciEngine::Score(
    const std::string& fen, const std::vector<std::string>& moves, int depth,
    int max_ms) {
  // One line per move, so that every move gets a score.
  if (!Send("setoption name MultiPV value " + std::to_string(moves.size())) ||
      !Send("position fen " + fen) ||
      !Send("go depth " + std::to_string(depth) + " movetime " +
            std::to_string(max_ms))) {
    return std::nullopt;
  }
  // Each info line with a score gives the first move of a line and its score;
  // later lines are deeper, so the last one for a move counts. Bounds are
  // skipped.
  std::unordered_map<std::string, int> scores;
  std::string line;
  while (true) {
    if (!ReadLine(&line)) return std::nullopt;
    if (line.rfind("bestmove", 0) == 0) break;
    if (line.rfind("info", 0) != 0) continue;
    std::istringstream iss(line);
    std::string token, kind, move;
    int value = 0;
    bool has_score = false, bound = false;
    while (iss >> token) {
      if (token == "score") {
        iss >> kind >> value;
        has_score = true;
      } else if (token == "lowerbound" || token == "upperbound") {
        bound = true;
      } else if (token == "pv") {
        iss >> move;
        break;
      }
    }
    if (!has_score || bound || move.empty()) continue;
    scores[move] = kind == "mate" ? (value > 0 ? kMateCp : -kMateCp) : value;
  }
  std::vector<int> result;
  result.reserve(moves.size());
  for (const std::string& move : moves) {
    const auto it = scores.find(move);
    result.push_back(it == scores.end() ? -kMateCp : it->second);
  }
  return result;
}

}  // namespace classic
}  // namespace lczero
