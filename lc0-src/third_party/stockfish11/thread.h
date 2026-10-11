/*
  Stand-in for Stockfish 11's thread.h, for Lc0's opponent model (see
  README.md). The evaluation only needs a thread's pawn and material hash
  tables, its node counter and its contempt.
*/

#ifndef THREAD_H_INCLUDED
#define THREAD_H_INCLUDED

#include <atomic>
#include <cstdint>

#include "material.h"
#include "pawns.h"
#include "types.h"

class Thread {
 public:
  Pawns::Table pawnsTable;
  Material::Table materialTable;
  std::atomic<uint64_t> nodes{0};
  Score contempt = SCORE_ZERO;
};

#endif  // #ifndef THREAD_H_INCLUDED
