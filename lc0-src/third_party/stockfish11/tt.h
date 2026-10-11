/*
  Stand-in for Stockfish 11's tt.h, for Lc0's opponent model (see README.md).
  Position::do_null_move() prefetches a transposition table entry; there is
  no transposition table here, so it prefetches a dummy one.
*/

#ifndef TT_H_INCLUDED
#define TT_H_INCLUDED

#include "types.h"

struct TTEntry {
  Key key;
};

struct TranspositionTable {
  TTEntry* first_entry(Key) { return &entry; }
  TTEntry entry{};
};

inline TranspositionTable TT;

#endif  // #ifndef TT_H_INCLUDED
