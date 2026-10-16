from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROMO = {None: 0, chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4}


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        return F.relu(x + self.b2(self.c2(F.relu(self.b1(self.c1(x))))))


class ChessNet(nn.Module):
    def __init__(self, channels, blocks, dim):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(17, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.trunk = nn.Sequential(*[Block(channels) for _ in range(blocks)])
        self.q = nn.Linear(channels, dim)
        self.k = nn.Linear(channels, dim)
        self.promo = nn.Linear(channels, 4)
        self.dim = dim

    def forward(self, board, castle, ep):
        n = board.shape[0]
        x = F.one_hot(board.long(), 13)[..., 1:].float().view(n, 8, 8, 12).permute(0, 3, 1, 2)
        c = ((castle.long().unsqueeze(1) >> torch.arange(4, device=board.device)) & 1).float()
        c = c.view(n, 4, 1, 1).expand(n, 4, 8, 8)
        e = F.one_hot(ep.long(), 65)[:, 1:].float().view(n, 1, 8, 8)
        h = self.trunk(self.stem(torch.cat([x, c, e], 1)))
        t = h.flatten(2).transpose(1, 2)
        return self.q(t) @ self.k(t).transpose(1, 2) / self.dim ** 0.5, self.promo(h.mean((2, 3)))


def default_device():
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.device("cuda")
    return torch.device("cpu")


def load(path, device):
    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    net = ChessNet(cfg["channels"], cfg["blocks"], cfg["dim"])
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()})
    return net.to(device).eval()


def encode(board, moves=None):
    flip = board.turn == chess.BLACK
    xor = 56 if flip else 0
    squares = np.zeros(64, np.int8)
    me, op = board.turn, not board.turn
    for pt in range(1, 7):
        for color, off in ((me, 0), (op, 6)):
            bb = board.pieces_mask(pt, color)
            if bb:
                if flip:
                    bb = chess.flip_vertical(bb)
                bits = np.unpackbits(np.frombuffer(bb.to_bytes(8, "little"), np.uint8), bitorder="little")
                squares[bits.astype(bool)] = pt + off
    castle = (int(board.has_kingside_castling_rights(me)) | int(board.has_queenside_castling_rights(me)) << 1
              | int(board.has_kingside_castling_rights(op)) << 2 | int(board.has_queenside_castling_rights(op)) << 3)
    ep = 0
    if board.ep_square is not None and board.has_legal_en_passant():
        ep = (board.ep_square ^ xor) + 1
    if moves is None:
        moves = list(board.legal_moves)
    codes = np.array([((m.from_square ^ xor) * 64 + (m.to_square ^ xor)) * 5 + PROMO[m.promotion]
                      for m in moves], np.int64)
    return squares, castle, ep, codes, moves


def policy(net, encs, device):
    lens = np.array([len(e[3]) for e in encs])
    width = int(lens.max())
    mask = np.arange(width)[None, :] < lens[:, None]
    codes = np.zeros((len(encs), width), np.int64)
    for i, e in enumerate(encs):
        codes[i, :len(e[3])] = e[3]
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    board = t(np.stack([e[0] for e in encs]))
    castle = t(np.array([e[1] for e in encs], np.int8))
    ep = t(np.array([e[2] for e in encs], np.int8))
    codes, mask = t(codes), t(mask)
    with torch.no_grad(), torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
        scores, promo = net(board, castle, ep)
    promo = torch.cat([torch.zeros_like(promo[:, :1]), promo], 1).float()
    logits = scores.float().flatten(1).gather(1, codes // 5) + promo.gather(1, codes % 5)
    probs = torch.softmax(logits.masked_fill(~mask, -1e9), 1).cpu().numpy()
    return [probs[i, :n].astype(np.float64) for i, n in enumerate(lens)]
