import argparse
import math
import time

import chess

from . import network
from .uci import MODEL, move_time
from .search import Search

PIECES = {"P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
          "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚"}


def show(board, white_bottom, ascii_only):
    ranks = range(7, -1, -1) if white_bottom else range(8)
    files = range(8) if white_bottom else range(7, -1, -1)
    print()
    for r in ranks:
        row = []
        for f in files:
            p = board.piece_at(chess.square(f, r))
            row.append((p.symbol() if ascii_only else PIECES[p.symbol()]) if p else ".")
        print(" %d  %s" % (r + 1, " ".join(row)))
    print("    " + " ".join("abcdefgh"[f] for f in files))
    print()


def read_move(board):
    while True:
        text = input("your move: ").strip()
        if text in ("quit", "exit"):
            return None
        if text == "undo":
            return "undo"
        for parse in (board.parse_san, board.parse_uci):
            try:
                return parse(text)
            except ValueError:
                pass
        print("  not a legal move (e.g. e4, Nf3, O-O, e7e8q) - 'undo' or 'quit' also work")


def clock_text(ms):
    s = max(ms, 0) / 1000
    return "%d:%04.1f" % (s // 60, s % 60) if s < 20 else "%d:%02d" % (s // 60, s % 60)


def measure_speed(search):
    board = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 9")
    search.run(board, 32)
    search.new_game()
    search.cache.clear()
    start = time.perf_counter()
    search.run(board, 256)
    search.new_game()
    search.cache.clear()
    return 256 / max(time.perf_counter() - start, 1e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--black", action="store_true", help="you play black")
    ap.add_argument("--steps", type=int, default=800, help="search steps per move (without a clock)")
    ap.add_argument("--time", type=float, help="minutes per side, e.g. 10 (rapid) or 1 (bullet)")
    ap.add_argument("--inc", type=float, default=0, help="seconds added after each move")
    ap.add_argument("--ascii", action="store_true", help="letters instead of chess symbols")
    args = ap.parse_args()

    device = network.default_device()
    search = Search(network.load(MODEL, device), device)
    board = chess.Board()
    you = chess.BLACK if args.black else chess.WHITE
    timed = args.time is not None
    clock = {chess.WHITE: args.time * 60000, chess.BLACK: args.time * 60000} if timed else None
    inc = args.inc * 1000
    speed = measure_speed(search) if timed else 0
    flagged = None

    while not board.is_game_over(claim_draw=True):
        show(board, you == chess.WHITE, args.ascii)
        if timed:
            print("clock: you %s | CareChess %s" % (clock_text(clock[you]), clock_text(clock[not you])))
        start = time.perf_counter()
        if board.turn == you:
            move = read_move(board)
            if move is None:
                return
            if move == "undo":
                if timed:
                    print("  no undo with a clock")
                elif len(board.move_stack) >= 2:
                    board.pop()
                    board.pop()
                continue
        else:
            steps = args.steps
            if timed:
                think = move_time(clock[board.turn], inc) - 50
                steps = max(16, int(speed * max(think, 10) / 1000))
            move, value = search.run(board, steps)
            secs = time.perf_counter() - start
            if timed and steps >= 64:
                speed = 0.7 * speed + 0.3 * steps / secs
            v = value if board.turn == chess.WHITE else -value
            pawns = round(4 * math.atanh(max(-0.999, min(0.999, v))), 2) + 0.0
            print("CareChess: %s   (%.1fs, %d steps, eval %+.2f)" % (board.san(move), secs, steps, pawns))
        if timed:
            clock[board.turn] -= (time.perf_counter() - start) * 1000
            if clock[board.turn] <= 0:
                flagged = board.turn
                break
            clock[board.turn] += inc
        board.push(move)

    show(board, you == chess.WHITE, args.ascii)
    if flagged is not None:
        print("%s lost on time" % ("you" if flagged == you else "CareChess"))
        print("game over:", "0-1" if flagged == chess.WHITE else "1-0")
    else:
        print("game over:", board.result(claim_draw=True))
