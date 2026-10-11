/*
  Stand-in for Stockfish 11's uci.h, for Lc0's opponent model (see README.md).
  Position only needs square names.
*/

#ifndef UCI_H_INCLUDED
#define UCI_H_INCLUDED

#include <string>

#include "types.h"

namespace UCI {

inline std::string square(Square s) {
  return std::string{char('a' + file_of(s)), char('1' + rank_of(s))};
}

}  // namespace UCI

#endif  // #ifndef UCI_H_INCLUDED
