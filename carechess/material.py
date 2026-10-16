import chess
import numpy as np

VALUES = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 310, chess.ROOK: 500, chess.QUEEN: 900}
MATE = 100000


def balance(board):
    return sum(v * (len(board.pieces(pt, board.turn)) - len(board.pieces(pt, not board.turn)))
               for pt, v in VALUES.items())


def captures_first(board, moves):
    def key(m):
        if board.is_en_passant(m):
            victim = 100
        else:
            p = board.piece_at(m.to_square)
            victim = VALUES.get(p.piece_type, 0) if p else 0
        attacker = VALUES.get(board.piece_at(m.from_square).piece_type, 0)
        return -(victim * 10 - attacker // 100 + (VALUES[m.promotion] * 10 if m.promotion else 0))
    return sorted(moves, key=key)


def quiescence(board, alpha, beta, ply=0, depth=6):
    if board.is_check():
        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply
        best = -MATE
    else:
        best = balance(board)
        if best >= beta or depth <= 0:
            return best
        alpha = max(alpha, best)
        moves = [m for m in board.generate_legal_moves() if board.is_capture(m) or m.promotion]
    for m in captures_first(board, moves):
        board.push(m)
        score = -quiescence(board, -beta, -alpha, ply + 1, depth - 1)
        board.pop()
        if score > best:
            best = score
            alpha = max(alpha, score)
            if score >= beta:
                break
    return best


def value(board):
    return float(np.tanh(quiescence(board, -MATE - 1, MATE + 1) / 400.0))
