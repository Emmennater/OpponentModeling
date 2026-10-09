#!/bin/bash
# Checks that a search repeated after 'ucinewgame' searches exactly the same
# number of nodes, at several node limits. With opponent modeling enabled, this
# only passes if 'ucinewgame' resets all of the opponent model's state: its
# hash table, histories and filter cache.
#
# Based on stockfish/tests/reprosearch.sh, which needs 'expect'. This version
# only needs bash.
#
# Usage: tests/reprosearch.sh [opponent depth]   (default: the engine's default)

ENGINE="$(dirname "$0")/../stockfish/src/stockfish.exe"
[ -x "$ENGINE" ] || ENGINE="$(dirname "$0")/../stockfish/src/stockfish"

OPTIONS=""
if [ -n "$1" ]; then
  OPTIONS="setoption name Opponent Depth value $1"
fi

FAILED=0

for i in $(seq 1 20); do
  nodes=$((100 * 3**i / 2**i))

  # Each 'go' waits for the previous search to finish. The final 'ucinewgame'
  # waits for the last one, so 'quit' does not cut it short.
  counts=$(printf '%s\n' \
    "$OPTIONS" \
    "ucinewgame" "position startpos" "go nodes $nodes" \
    "position startpos moves e2e4 e7e6" "go nodes $nodes" \
    "ucinewgame" "position startpos" "go nodes $nodes" \
    "position startpos moves e2e4 e7e6" "go nodes $nodes" \
    "ucinewgame" "quit" |
    "$ENGINE" 2>&1 |
    awk '/^info depth/ { for (k = 1; k <= NF; k++) if ($k == "nodes") last = $(k + 1) }
         /^bestmove/  { printf "%s ", last }')

  set -- $counts
  if [ $# -eq 4 ] && [ "$1" = "$3" ] && [ "$2" = "$4" ]; then
    echo "nodes $nodes: ok ($counts)"
  else
    echo "nodes $nodes: FAILED ($counts)"
    FAILED=1
  fi
done

if [ $FAILED -eq 0 ]; then
  echo "reprosearch passed"
else
  echo "reprosearch FAILED"
  exit 1
fi
