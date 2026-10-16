import math
import sys
import threading
import time
from pathlib import Path

import chess

from . import network
from .search import Search

NAME = "CareChess"
MODEL = Path(__file__).resolve().parent.parent / "models" / "human.pt"


def move_time(left_ms, inc_ms=0, moves_to_go=0):
    return min(left_ms / (moves_to_go or 30) + 0.8 * inc_ms, left_ms / 5)


class Engine:
    def __init__(self):
        self.board = chess.Board()
        self.search = None
        self.steps = 0
        self.overhead = 100
        self.speed = 300.0
        self.stop = threading.Event()
        self.thread = None

    def send(self, line):
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def load(self):
        if self.search is None:
            device = network.default_device()
            self.search = Search(network.load(MODEL, device), device)

    def budget(self, go):
        if go.get("nodes"):
            return go["nodes"]
        if self.steps:
            return self.steps
        if go.get("movetime"):
            ms = go["movetime"] - self.overhead
        else:
            side = "w" if self.board.turn else "b"
            left = go.get(side + "time")
            if left is None:
                return 800
            ms = move_time(left, go.get(side + "inc", 0), go.get("movestogo", 0)) - self.overhead
        return int(max(16, self.speed * max(ms, 10) / 1000))

    def think(self, go):
        start = time.perf_counter()
        if go.get("infinite"):
            done = 0
            while True:
                move, value = self.search.run(self.board, 512)
                done += 512
                self.info(move, value, done, start)
                if self.stop.is_set():
                    break
        else:
            steps = self.budget(go)
            move, value = self.search.run(self.board, steps)
            secs = time.perf_counter() - start
            if steps >= 128 and secs > 0.05:
                self.speed = 0.7 * self.speed + 0.3 * steps / secs
            self.info(move, value, steps, start)
        self.send("bestmove " + move.uci())

    def info(self, move, value, steps, start):
        ms = max(1, int((time.perf_counter() - start) * 1000))
        cp = int(111.7 * math.tan(1.562 * max(-0.999, min(0.999, value))))
        pv = self.search.pv(move)
        self.send("info depth %d nodes %d nps %d time %d score cp %d pv %s"
                  % (len(pv), steps, steps * 1000 // ms, ms, cp, " ".join(m.uci() for m in pv)))

    def wait(self):
        if self.thread:
            self.thread.join()
            self.thread = None

    def position(self, args):
        if args[0] == "startpos":
            board, rest = chess.Board(), args[1:]
        elif args[0] == "fen":
            board, rest = chess.Board(" ".join(args[1:7])), args[7:]
        else:
            return
        for uci in rest[1:] if rest[:1] == ["moves"] else []:
            board.push_uci(uci)
        self.board = board

    def go(self, args):
        go, i = {}, 0
        while i < len(args):
            if args[i] in ("wtime", "btime", "winc", "binc", "movestogo", "movetime", "nodes", "depth"):
                go[args[i]] = int(args[i + 1])
                i += 2
            else:
                go[args[i]] = True
                i += 1
        self.load()
        self.stop.clear()
        self.thread = threading.Thread(target=self.think, args=(go,), daemon=True)
        self.thread.start()

    def setoption(self, args):
        if "name" not in args:
            return
        i = args.index("name")
        j = args.index("value") if "value" in args else len(args)
        name, value = " ".join(args[i + 1:j]), " ".join(args[j + 1:])
        if name == "Steps":
            self.steps = int(value or 0)
        elif name == "Move Overhead":
            self.overhead = int(value or 0)

    def loop(self):
        for line in sys.stdin:
            cmd, *args = line.split() or [""]
            if cmd == "uci":
                self.send("id name " + NAME)
                self.send("option name Steps type spin default 0 min 0 max 1000000")
                self.send("option name Move Overhead type spin default 100 min 0 max 5000")
                self.send("uciok")
            elif cmd == "isready":
                self.load()
                self.send("readyok")
            elif cmd == "setoption":
                self.setoption(args)
            elif cmd == "ucinewgame":
                self.wait()
                if self.search:
                    self.search.new_game()
            elif cmd == "position" and args:
                self.wait()
                self.position(args)
            elif cmd == "go":
                self.wait()
                self.go(args)
            elif cmd == "stop":
                self.stop.set()
                self.wait()
            elif cmd == "quit":
                break
        self.stop.set()
        self.wait()


def main():
    Engine().loop()
