import math

import numpy as np

from . import material, network

C_PUCT = 1.5
FPU = 0.5
C_VISIT, C_SCALE = 50.0, 0.1
CACHE_MAX = 250_000


class Node:
    __slots__ = ("moves", "P", "N", "W", "children", "value", "term")

    def __init__(self):
        self.moves = None
        self.term = None


def expand(node, priors, value):
    node.P, node.value = priors, value
    node.N = np.zeros(len(priors))
    node.W = np.zeros(len(priors))
    node.children = [None] * len(priors)


def game_over_value(board, moves):
    if not moves:
        return -1.0 if board.is_check() else 0.0
    if board.is_insufficient_material() or board.halfmove_clock >= 100 or board.is_repetition(2):
        return 0.0
    return None


def pick_child(node):
    n = node.N
    q = node.W / np.maximum(n, 1.0)
    q[n == 0] = node.value - FPU
    return int((q + C_PUCT * math.sqrt(n.sum() + 1) * node.P / (1.0 + n)).argmax())


def backup(path, value):
    v = -value
    for node, i in reversed(path):
        node.N[i] += 1
        node.W[i] += v
        v = -v


def remove_virtual_loss(path):
    for node, i in path:
        node.N[i] -= 1
        node.W[i] += 1


def sigma(root):
    n = root.N
    seen = n > 0
    q = np.where(seen, root.W / np.maximum(n, 1), 0.0)
    total = n.sum()
    mixed = root.value
    if total > 0:
        mixed = (root.value + total * (root.P[seen] * q[seen]).sum() / max(root.P[seen].sum(), 1e-12)) / (1 + total)
    q = np.where(seen, q, mixed)
    q = (q - q.min()) / max(q.max() - q.min(), 1e-8)
    return (C_VISIT + n.max()) * C_SCALE * q


class Search:
    def __init__(self, net, device):
        self.net, self.device = net, device
        self.cache = {}
        self.tree = {}

    def new_game(self):
        self.tree = {}

    def priors(self, encs):
        keys = [e[5] for e in encs]
        missing = [i for i, k in enumerate(keys) if k not in self.cache]
        if missing:
            if len(self.cache) > CACHE_MAX:
                self.cache.clear()
            for i, p in zip(missing, network.policy(self.net, [encs[i] for i in missing], self.device)):
                self.cache[keys[i]] = p
        return [self.cache[k].copy() for k in keys]

    def reused_root(self, board):
        t = self.tree
        if t.get("root") is None:
            return None
        if len(board.move_stack) == t["plies"] and board.fen() == t["fen"]:
            return t["root"]
        if len(board.move_stack) != t["plies"] + 2:
            return None
        before = board.copy()
        reply = before.pop()
        ours = before.pop()
        root = t["root"]
        if before.fen() != t["fen"] or ours not in root.moves:
            return None
        child = root.children[root.moves.index(ours)]
        if child is None or child.moves is None or reply not in child.moves:
            return None
        node = child.children[child.moves.index(reply)]
        if node is None or node.moves is None or node.term not in (None, False):
            return None
        return node

    def descend(self, root, board, first):
        node, path, i = root, [], first
        while True:
            board.push(node.moves[i])
            path.append((node, i))
            node.N[i] += 1
            node.W[i] -= 1
            if node.children[i] is None:
                node.children[i] = Node()
            node = node.children[i]
            moves = None
            if node.term is None:
                moves = list(board.legal_moves)
                v = game_over_value(board, moves)
                node.term = False if v is None else v
            if node.term is not False or node.moves is None:
                leaf = None
                if node.term is False:
                    leaf = network.encode(board, moves) + (board._transposition_key(), material.value(board))
                for _ in path:
                    board.pop()
                return path, node, (None if node.term is False else node.term), leaf
            i = pick_child(node)

    def run(self, board, steps, m=16, batch=64):
        b = board.copy()
        root = self.reused_root(board)
        if root is None:
            root = Node()
            enc = network.encode(b) + (b._transposition_key(),)
            root.moves = enc[4]
            expand(root, self.priors([enc])[0], material.value(b))
        logits = np.log(np.maximum(root.P, 1e-12))
        budget = steps - 1
        if budget > 0 and len(root.moves) > 1:
            k = min(m, len(root.moves))
            cands = list(np.argsort(-logits)[:k])
            rounds = max(1, math.ceil(math.log2(k)))
            used = 0
            while used < budget:
                walks = [a for _ in range(max(1, budget // (rounds * len(cands)))) for a in cands][:budget - used]
                for s in range(0, len(walks), batch):
                    pending = {}
                    for a in walks[s:s + batch]:
                        path, node, final, leaf = self.descend(root, b, a)
                        if final is not None:
                            remove_virtual_loss(path)
                            backup(path, final)
                        else:
                            pending.setdefault(id(node), (node, leaf, []))[2].append(path)
                    if pending:
                        items = list(pending.values())
                        for (node, leaf, paths), p in zip(items, self.priors([x[1] for x in items])):
                            if node.moves is None:
                                node.moves = leaf[4]
                                expand(node, p, leaf[6])
                            for path in paths:
                                remove_virtual_loss(path)
                                backup(path, leaf[6])
                used += len(walks)
                if len(cands) > 1:
                    score = logits + sigma(root)
                    cands = sorted(cands, key=lambda a: -score[a])[:(len(cands) + 1) // 2]
            score = logits + sigma(root)
            best = int(max(cands, key=lambda a: score[a]))
        else:
            best = int(np.argmax(root.P))
        self.tree = {"root": root, "plies": len(board.move_stack), "fen": board.fen()}
        value = float(root.W.sum() / root.N.sum()) if root.N.sum() > 0 else root.value
        return root.moves[best], value

    def pv(self, first, length=16):
        line, root = [first], self.tree.get("root")
        if root is None or first not in root.moves:
            return line
        node = root.children[root.moves.index(first)]
        while node is not None and node.moves is not None and node.N.sum() > 0 and len(line) < length:
            i = int(node.N.argmax())
            line.append(node.moves[i])
            node = node.children[i]
        return line
