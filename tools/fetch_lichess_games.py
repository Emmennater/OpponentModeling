"""Streams full games from a Lichess monthly database file.

The monthly files (database.lichess.org) are about 30 GB of compressed PGN.
This reads one from the start, decompressing as it downloads, keeps the games
that match the filters, and stops once it has enough, so only a small part of
the file is ever downloaded and nothing large is written to disk.

Output has the format of lichess_games.tsv: one game per line, tab-separated
white rating, black rating, result, and the moves in SAN separated by spaces,
here for the whole game.

Needs the zstandard package (pip install zstandard).

Usage:
    python tools/fetch_lichess_games.py full_games.tsv --month 2026-08 --games 3000
"""

import argparse
import io
import re
import sys
import time
import urllib.request

import zstandard

URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
HEADER = re.compile(r'^\[(\w+) "(.*)"\]$')
COMMENT = re.compile(r"\{[^}]*\}")
MOVE_NUMBER = re.compile(r"^\d+\.+$")
RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}


def games(stream):
    """Yields (headers, movetext) for each game in a PGN text stream."""
    headers, movetext = {}, []
    for line in stream:
        line = line.strip()
        if not line:
            continue
        m = HEADER.match(line)
        if m:
            if movetext:
                yield headers, " ".join(movetext)
                headers, movetext = {}, []
            headers[m.group(1)] = m.group(2)
        else:
            movetext.append(line)
    if movetext:
        yield headers, " ".join(movetext)


def san_moves(movetext):
    """The moves of a movetext, without comments, move numbers, annotations
    or the result."""
    tokens = COMMENT.sub(" ", movetext).split()
    return [t.rstrip("?!") for t in tokens if not MOVE_NUMBER.match(t) and t not in RESULTS]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output")
    parser.add_argument("--month", default="2026-08", help="database month, YYYY-MM (default 2026-08)")
    parser.add_argument("--games", type=int, default=3000, help="games to keep (default 3000)")
    parser.add_argument("--min-rating", type=int, default=1200, help="both players at least this (default 1200)")
    parser.add_argument("--max-rating", type=int, default=1399, help="both players at most this (default 1399)")
    parser.add_argument("--speeds", default="Blitz,Rapid", help="Lichess speeds kept (default Blitz,Rapid)")
    parser.add_argument("--min-plies", type=int, default=20, help="shortest game kept (default 20)")
    args = parser.parse_args()

    speeds = {f"Rated {s.strip()} game" for s in args.speeds.split(",")}
    request = urllib.request.Request(URL.format(month=args.month), headers={"User-Agent": "OpponentModeling"})
    start, read, kept = time.time(), 0, 0

    with urllib.request.urlopen(request) as response, open(args.output, "w", encoding="utf-8") as out:
        counted = CountingReader(response)
        reader = zstandard.ZstdDecompressor().stream_reader(counted)
        for headers, movetext in games(io.TextIOWrapper(reader, encoding="utf-8")):
            read += 1
            if headers.get("Event") not in speeds:
                continue
            try:
                white, black = int(headers["WhiteElo"]), int(headers["BlackElo"])
            except (KeyError, ValueError):
                continue
            if not all(args.min_rating <= r <= args.max_rating for r in (white, black)):
                continue
            moves = san_moves(movetext)
            if len(moves) < args.min_plies:
                continue
            out.write(f"{white}\t{black}\t{headers.get('Result', '*')}\t{' '.join(moves)}\n")
            kept += 1
            if kept >= args.games:
                break

    print(f"Kept {kept} of {read} games, downloading {counted.bytes / 1e6:.0f} MB in {time.time() - start:.0f} s")


class CountingReader:
    """A file wrapper that counts the bytes read through it."""

    def __init__(self, raw):
        self.raw, self.bytes = raw, 0

    def read(self, size=-1):
        data = self.raw.read(size)
        self.bytes += len(data)
        return data


if __name__ == "__main__":
    sys.exit(main())
