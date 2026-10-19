"""Measures the engine's search speed with each feature, against the Lc0 release.

For each configuration it starts the engine, loads the networks, warms up, then
searches each opening position to a fixed node count with smart pruning off, and
records:
- nodes per second, overall and over the course of the search;
- the CPU time of Lc0 and its Stockfish helpers, as cores busy on average;
- the GPU's average utilization, sampled by nvidia-smi;
- the time from launch to the engine being ready (network loading, and Maia-3's
  CUDA graphs).

Results are saved to a JSON file as they come, so an interrupted run resumes.

Usage:
    python tools/bench_speed.py --out bench.json --nodes 300000
    python tools/bench_speed.py --out bench.json --only release,build,om-full
    python tools/bench_speed.py --list
"""

import argparse
import json
import queue
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RELEASE = ROOT / "lc0" / "bin" / "lc0.exe"
BUILD = ROOT / "lc0-src" / "build" / "lc0.exe"
NET = ROOT / "lc0" / "bin" / "791556.pb.gz"
FINETUNED = ROOT / "training" / "nets" / "net09.pb.gz"
NETS = ROOT / "lc0" / "nets"
STOCKFISH = ROOT.parent / "StockfishTrapFinder" / "stockfish" / "stockfish.exe"

POSITIONS = {
    "start": "",
    "1.e4 c5": "e2e4 c7c5",
    "Belgrade 5": "e2e4 e7e5 g1f3 b8c6 b1c3 g8f6 d2d4 e5d4",
    "QGD": "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",
}

COMMON = {"SmartPruningFactor": 0}

# An opponent model with every extra off: the cost of the model alone.
OM_BARE = {
    "OpponentModelRating": 1500, "OpponentModelOwnRating": 1500, "OpponentModelTemperature": 1,
    "OpponentTactics": 0, "OpponentMinProbability": 0, "OpponentEngineMix": 0,
    "OpponentRootCandidates": 0, "OpponentDeepTactics": 0, "OpponentCaution": 0,
    "OpponentCautionAhead": 0, "OpponentHorizon": 0, "OpponentWideCandidates": 0,
}

# The user's playing settings (En Croissant, 2026-10-15).
OM_FULL = {
    "OpponentModelWeights": str(NETS / "maia3-23m-fp16.onnx"),
    "OpponentModelRating": 1500, "OpponentModelOwnRating": 1500, "OpponentModelTemperature": 1,
    "OpponentTactics": 0, "OpponentMinProbability": 0, "OpponentEngineMix": 0,
    "OpponentRootCandidates": 0,
    "OpponentDeepTactics": 1, "OpponentDeepTacticsDepth": 4, "OpponentDeepTacticsVisits": 1000,
    "OpponentDeepTacticsThreads": 8, "OpponentStockfishFloor": True,
    "OpponentStockfishPath": str(STOCKFISH),
    "CPuct": 2.5, "PolicyTemperature": 1.359, "OpponentBestLean": 1, "OpponentDiscount": 1,
    "OpponentCaution": 0.15, "OpponentRefutationProbability": 0.05, "OpponentCautionAhead": 0.25,
    "OpponentHorizon": 10, "OpponentWideVisits": 10000, "OpponentWideShare": 0.5,
    "OpponentWideCandidates": 0,
    "WeightsFile": str(FINETUNED), "IdentityPlane": True,
    "ScoreType": "win_percentage", "MultiPV": 3,
    "MovesLeftMaxEffect": 0.5, "MovesLeftThreshold": 0, "MovesLeftSlope": 0.02,
    "MovesLeftScaledFactor": 0.15, "MovesLeftQuadraticFactor": 0.85,
}


def om(model, **extra):
    return {**OM_BARE, "OpponentModelWeights": str(NETS / model), **extra}


def full(**changes):
    return {**OM_FULL, **changes}


# name: (binary, options)
CONFIGS = {
    # Plain Lc0
    "release": (RELEASE, {}),
    "build": (BUILD, {}),
    "release-t1": (RELEASE, {"Threads": 1}),
    "build-t1": (BUILD, {"Threads": 1}),
    "release-t4": (RELEASE, {"Threads": 4}),
    "build-t4": (BUILD, {"Threads": 4}),
    "build-net09": (BUILD, {"WeightsFile": str(FINETUNED), "IdentityPlane": True}),
    # The opponent model alone
    "om-maia1": (BUILD, om("maia-1500.pb.gz")),
    "om-maia3-5m-fp16": (BUILD, om("maia3-5m-fp16.onnx")),
    "om-maia3-23m-fp16": (BUILD, om("maia3-23m-fp16.onnx")),
    "om-maia3-23m-fp32": (BUILD, om("maia3-23m.onnx")),
    "om-maia3-79m-fp16": (BUILD, om("maia3-79m-fp16.onnx")),
    # The build's defaults for the extras, with Maia3-23M fp16
    "om-defaults": (BUILD, {"OpponentModelWeights": str(NETS / "maia3-23m-fp16.onnx"),
                            "OpponentModelRating": 1500, "OpponentModelOwnRating": 1500}),
    "om-bare+tactics": (BUILD, om("maia3-23m-fp16.onnx", OpponentTactics=0.25)),
    "om-bare+rootcand": (BUILD, om("maia3-23m-fp16.onnx", OpponentRootCandidates=8)),
    "om-bare+wide": (BUILD, om("maia3-23m-fp16.onnx", OpponentWideCandidates=4)),
    # The user's settings, then each feature taken out
    "om-full": (BUILD, full()),
    "om-full-t1": (BUILD, full(Threads=1)),
    "om-full-t4": (BUILD, full(Threads=4)),
    "om-full-no-closer-look": (BUILD, full(OpponentDeepTactics=0, OpponentStockfishFloor=False)),
    "om-full-no-floor": (BUILD, full(OpponentStockfishFloor=False)),
    "om-full-sf2threads": (BUILD, full(OpponentDeepTacticsThreads=2)),
    "om-full-no-caution": (BUILD, full(OpponentCaution=0, OpponentCautionAhead=0)),
    "om-full-no-horizon": (BUILD, full(OpponentHorizon=0)),
    "om-full-no-movesleft": (BUILD, full(MovesLeftMaxEffect=0)),
    "om-full-multipv1": (BUILD, full(MultiPV=1)),
    "om-full-791556": (BUILD, full(WeightsFile=str(NET), IdentityPlane=False)),
}


def cpu_seconds(pid):
    """CPU time (user + kernel) of a process and its direct children."""
    script = (f"Get-CimInstance Win32_Process | Where-Object {{ $_.ProcessId -eq {pid} -or "
              f"$_.ParentProcessId -eq {pid} }} | ForEach-Object {{ \"$($_.Name) "
              f"$($_.KernelModeTime + $_.UserModeTime)\" }}")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                         capture_output=True, text=True).stdout
    lc0 = helpers = 0.0
    for line in out.split("\n"):
        parts = line.split()
        if len(parts) == 2:
            secs = int(parts[1]) / 1e7
            if parts[0].lower().startswith("lc0"):
                lc0 += secs
            elif parts[0].lower() != "conhost.exe":
                helpers += secs
    return lc0, helpers


class GpuSampler:
    def __init__(self):
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits",
             "-lms", "200"], stdout=subprocess.PIPE, text=True)
        self.samples = []
        self.active = False
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            if self.active:
                try:
                    util, mem = (float(x) for x in line.split(","))
                    self.samples.append((util, mem))
                except ValueError:
                    pass

    def start(self):
        self.samples = []
        self.active = True

    def stop(self):
        self.active = False
        s = self.samples
        return (statistics.mean(u for u, _ in s) if s else None,
                max(m for _, m in s) if s else None)

    def close(self):
        self.proc.kill()


class Engine:
    def __init__(self, binary, options):
        self.t0 = time.perf_counter()
        self.proc = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1,
                                     cwd=str(binary.parent))
        self.lines = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self.send("uci")
        self.wait("uciok")
        opts = {**COMMON, "WeightsFile": str(NET), **options}
        for k, v in opts.items():
            v = str(v).lower() if isinstance(v, bool) else v
            self.send(f"setoption name {k} value {v}")
        self.send("isready")
        self.wait("readyok", timeout=300)
        self.go("", 2000)  # loads the networks and records Maia-3's graphs
        self.ready_secs = time.perf_counter() - self.t0

    def _read(self):
        for line in self.proc.stdout:
            self.lines.put((time.perf_counter(), line.strip()))

    def send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def wait(self, token, timeout=600):
        end = time.perf_counter() + timeout
        out = []
        while True:
            t, line = self.lines.get(timeout=max(0.1, end - time.perf_counter()))
            out.append((t, line))
            if line.startswith(token):
                return out
            if "error" in line.lower() or "exception" in line.lower():
                print("   engine:", line, file=sys.stderr)

    def go(self, moves, nodes):
        self.send("ucinewgame")
        self.send(f"position startpos moves {moves}" if moves else "position startpos")
        self.send("isready")
        self.wait("readyok")
        t0 = time.perf_counter()
        self.send(f"go nodes {nodes}")
        out = self.wait("bestmove", timeout=3600)
        wall = out[-1][0] - t0
        series = []  # (engine time ms, nodes)
        for _, line in out:
            if line.startswith("info") and " nodes " in line and " time " in line:
                tok = line.split()
                series.append((int(tok[tok.index("time") + 1]), int(tok[tok.index("nodes") + 1])))
        return wall, series, out[-1][1]

    def close(self):
        try:
            self.send("quit")
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def nps_by_phase(series, nodes):
    """Nodes per second over successive fifths of the search."""
    marks, out, prev = [nodes * i / 5 for i in range(1, 6)], [], (0, 0)
    for m in marks:
        pt = next(((t, n) for t, n in series if n >= m), series[-1] if series else None)
        if pt and pt[0] > prev[0]:
            out.append(round((pt[1] - prev[1]) / (pt[0] - prev[0]) * 1000))
            prev = pt
    return out


def run_config(name, nodes, gpu):
    binary, options = CONFIGS[name]
    eng = Engine(binary, options)
    res = {"ready_secs": round(eng.ready_secs, 2), "positions": {}}
    pid = eng.proc.pid
    for pos, moves in POSITIONS.items():
        c0 = cpu_seconds(pid)
        gpu.start()
        wall, series, best = eng.go(moves, nodes)
        util, mem = gpu.stop()
        c1 = cpu_seconds(pid)
        done = series[-1][1] if series else nodes
        res["positions"][pos] = {
            "nodes": done, "wall": round(wall, 3), "nps": round(done / wall),
            "nps_by_fifth": nps_by_phase(series, done),
            "lc0_cores": round((c1[0] - c0[0]) / wall, 2),
            "helper_cores": round((c1[1] - c0[1]) / wall, 2),
            "gpu_util": round(util, 1) if util is not None else None,
            "gpu_mem_mb": mem, "bestmove": best.split()[1] if len(best.split()) > 1 else "",
        }
        p = res["positions"][pos]
        print(f"   {pos:<12} {p['nps']:>8,} nps  {wall:6.1f} s  lc0 {p['lc0_cores']:5.2f} cores  "
              f"sf {p['helper_cores']:5.2f}  gpu {p['gpu_util']}%  by fifth {p['nps_by_fifth']}",
              flush=True)
    eng.close()
    ps = res["positions"].values()
    res["nodes"] = sum(p["nodes"] for p in ps)
    res["wall"] = round(sum(p["wall"] for p in ps), 2)
    res["nps"] = round(res["nodes"] / res["wall"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench.json")
    ap.add_argument("--nodes", type=int, default=300000)
    ap.add_argument("--only", default="", help="comma-separated configs (default: all)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--redo", action="store_true", help="rerun configs already in --out")
    args = ap.parse_args()
    if args.list:
        print("\n".join(CONFIGS))
        return
    names = [n for n in args.only.split(",") if n] or list(CONFIGS)
    out = Path(args.out)
    data = json.loads(out.read_text()) if out.exists() else {}
    gpu = GpuSampler()
    try:
        for name in names:
            key = f"{name}@{args.nodes}"
            if key in data and not args.redo:
                print(f"{name}: done, {data[key]['nps']:,} nps")
                continue
            print(f"{name}:", flush=True)
            data[key] = run_config(name, args.nodes, gpu)
            print(f"   => {data[key]['nps']:,} nps overall, ready in {data[key]['ready_secs']} s",
                  flush=True)
            out.write_text(json.dumps(data, indent=1))
    finally:
        gpu.close()


if __name__ == "__main__":
    main()
