#!/usr/bin/env python3
# Run the same positions on pysayuri (raw_nn, the .pt weights) and the C++
# engines (sayuri-raw_nn 0, the .txt weights, fp32 without winograd by
# default), then compare the network outputs.
import argparse
import os
import shlex
import subprocess
import sys

import numpy as np

TORCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYSAYURI = os.path.join(TORCH_DIR, "pysayuri.py")
ENGINE_ARGS = "--no-winograd --no-fp16 -t 1"

# The smaller boards check the padding and mask path of the python network,
# which always runs on the 19x19 planes.
POSITIONS = [
    ("19x19 empty", 19, []),
    ("19x19 opening", 19, [
        "B Q16", "W D4", "B Q3", "W D16", "B R10", "W C10", "B K10", "W K4", "B O4"]),
    ("13x13 opening", 13, ["B D4", "W K10", "B K4", "W D10", "B G7"]),
    ("9x9 empty", 9, []),
    ("9x9 opening", 9, ["B E5", "W C3", "B G7", "W C7", "B G3", "W E3"]),
]

def make_commands(board_size, moves, raw_cmd):
    cmds = ["boardsize {}".format(board_size), "komi 7.5", "clear_board"]
    cmds += ["play {}".format(m) for m in moves]
    cmds += [raw_cmd, "quit"]
    return "\n".join(cmds) + "\n"

def read_grid(lines, idx, board_size):
    grid = list()
    for _ in range(board_size):
        grid.append([float(v) for v in lines[idx].split()])
        idx += 1
    return np.array(grid), idx

def run_python(pt, board_size, moves):
    proc = subprocess.run(
        [sys.executable, PYSAYURI, "-c", pt],
        input=make_commands(board_size, moves, "raw_nn"),
        capture_output=True, text=True, timeout=600)

    # The raw_nn of pysayuri prints the outputs to stderr.
    lines = [l.rstrip() for l in proc.stderr.splitlines()]
    out = dict()
    idx = 0
    while idx < len(lines):
        l = lines[idx]
        if l.startswith("winrate="):
            out["winrate"] = float(l.split()[1][:-1]) / 100
        elif l.startswith("drawrate="):
            out["drawrate"] = float(l.split()[1][:-1]) / 100
        elif l.startswith("score="):
            out["score"] = float(l.split()[1])
        elif l.startswith("policy="):
            out["policy"], idx = read_grid(lines, idx + 1, board_size)
            continue
        elif l.startswith("pass policy="):
            out["pass"] = float(l.split()[2])
        elif l.startswith("ownership="):
            out["ownership"], idx = read_grid(lines, idx + 1, board_size)
            continue
        idx += 1
    if len(out) != 6:
        raise Exception("the pysayuri output is broken:\n" + proc.stderr)
    return out

def run_cpp(engine, txt, engine_args, board_size, moves):
    proc = subprocess.run(
        [engine, "-w", txt] + shlex.split(engine_args),
        input=make_commands(board_size, moves, "sayuri-raw_nn 0"),
        capture_output=True, text=True, timeout=600)

    lines = [l.rstrip() for l in proc.stdout.splitlines()]
    lines = [l[2:] if l.startswith("= ") else l for l in lines]
    out = dict()
    idx = 0
    while idx < len(lines):
        l = lines[idx]
        if l.startswith("wdl winrate:"):
            out["winrate"] = float(l.split(":")[1])
        elif l.startswith("draw probability:"):
            out["drawrate"] = float(l.split(":")[1])
        elif l.startswith("final score:"):
            out["score"] = float(l.split(":")[1])
        elif l.startswith("probabilities:"):
            out["policy"], idx = read_grid(lines, idx + 1, board_size)
            continue
        elif l.startswith("pass probabilities:"):
            out["pass"] = float(l.split(":")[1])
        elif l.startswith("ownership:"):
            out["ownership"], idx = read_grid(lines, idx + 1, board_size)
            continue
        idx += 1
    if len(out) != 6:
        raise Exception("the C++ output is broken:\n" + proc.stdout + proc.stderr)
    return out

def is_saturated(out):
    # The saturated outputs are equal whether the network is correct or not.
    probs = np.append(out["policy"].ravel(), out["pass"])
    return out["winrate"] < 1e-3 or out["winrate"] > 1. - 1e-3 or \
               out["drawrate"] > 1. - 1e-3 or probs.max() > 0.99 or \
               np.abs(out["ownership"]).min() > 0.999

def compute_diffs(py, cc):
    return {
        "winrate" : abs(py["winrate"] - cc["winrate"]),
        "drawrate" : abs(py["drawrate"] - cc["drawrate"]),
        # The C++ engine prints the score with 6 significant digits.
        "score" : abs(py["score"] - cc["score"]) / max(1., abs(py["score"])),
        "pass" : abs(py["pass"] - cc["pass"]),
        "policy" : np.abs(py["policy"] - cc["policy"]).max(),
        "ownership" : np.abs(py["ownership"] - cc["ownership"]).max(),
    }

def main(args):
    engines = list()
    for e in args.engine:
        if not "=" in e:
            raise Exception("the engine should be <name>=<path>")
        engines.append(e.split("=", 1))

    all_pass = True
    for title, board_size, moves in POSITIONS:
        py = run_python(args.pt, board_size, moves)
        print("== {} | py winrate={:.6f} drawrate={:.6f} score={:.6f} pass={:.6f}".format(
                  title, py["winrate"], py["drawrate"], py["score"], py["pass"]))
        if is_saturated(py):
            all_pass = False
            print("   SATURATED, lower --noise or --weight-scale of gen_random_weights.py")
            continue

        for name, engine in engines:
            cc = run_cpp(engine, args.txt, args.engine_args, board_size, moves)
            diffs = compute_diffs(py, cc)
            ok = all(v <= args.tol for v in diffs.values())
            all_pass &= ok
            print("   {:10s} {} | ".format(name, "PASS" if ok else "FAIL") +
                  " ".join("{}={:.2e}".format(k, v) for k, v in diffs.items()))
            if args.dump and not ok:
                np.set_printoptions(precision=6, suppress=True, linewidth=200)
                print("py policy\n{}\ncc policy\n{}".format(py["policy"], cc["policy"]))
    print("ALL PASS" if all_pass else "SOME FAILED")
    return 0 if all_pass else 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", metavar="<string>", required=True,
                        help="The .pt weights for pysayuri.", type=str)
    parser.add_argument("--txt", metavar="<string>", required=True,
                        help="The .txt weights for the C++ engine.", type=str)
    parser.add_argument("--engine", metavar="<name=path>", action="append", required=True,
                        help="The C++ engine binary, can be repeated.", type=str)
    parser.add_argument("--engine-args", metavar="<string>", default=ENGINE_ARGS,
                        help="The arguments of the C++ engines.", type=str)
    parser.add_argument("--tol", metavar="<float>", default=1e-4,
                        help="The max acceptable difference.", type=float)
    parser.add_argument("--dump", default=False,
                        help="Dump the policy when it fails.", action="store_true")
    sys.exit(main(parser.parse_args()))
