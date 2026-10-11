/*
  Stand-in for Stockfish 11's syzygy/tbprobe.h, for Lc0's opponent model (see
  README.md). Position's debug printing probes tablebases when any are
  loaded; none ever are here.
*/

#ifndef TBPROBE_H
#define TBPROBE_H

class Position;

namespace Tablebases {

enum WDLScore { WDLLoss = -2, WDLBlessedLoss = -1, WDLDraw = 0, WDLCursedWin = 1, WDLWin = 2 };
enum ProbeState { FAIL = 0, OK = 1, CHANGE_STM = -1, ZEROING_BEST_MOVE = 2 };

inline int MaxCardinality = 0;

inline WDLScore probe_wdl(Position&, ProbeState* result) { *result = FAIL; return WDLDraw; }
inline int probe_dtz(Position&, ProbeState* result) { *result = FAIL; return 0; }

}  // namespace Tablebases

#endif
