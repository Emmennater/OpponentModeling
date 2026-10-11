# Stockfish 11, for the opponent model's quick tactical look

A subset of [Stockfish 11](https://github.com/official-stockfish/Stockfish/tree/sf_11) (GPL 3, see `Copying.txt`). It is used by `src/search/classic/opponent_tactics.cc`, which scores the modeled opponent's moves for the `OpponentTactics` option. Each move is scored by a depth-1 search: the move, then the capture sequences, judged by Stockfish 11's classical evaluation. Stockfish 11 is the last version whose evaluation is hand-written: material, piece placement, mobility, king safety and pawns, with no neural network.

The sources are Stockfish 11's own, for the board, move generation and evaluation, with these changes:
- **Stand-in headers:** `thread.h`, `tt.h`, `uci.h` and `syzygy/tbprobe.h` replace Stockfish's own. They provide only what the files below use: a thread's pawn and material tables, node counter and contempt; a dummy transposition table entry for a prefetch; square names; and tablebase probes that report no tablebases.
- **`pawns.h`:** the pawn hash table has 16,384 entries instead of 131,072, since every Lc0 search thread has one.
- **`material.cpp`:** `clamp` is called as `::clamp`, which is Stockfish's own, because it is ambiguous with `std::clamp` in C++20.
- **`misc.cpp`:** includes `<mutex>`, which Visual Studio's headers do not include through the others.

Built as the `stockfish11` static library in `meson.build`, with `USE_POPCNT`.
