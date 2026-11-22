"""Random phylogenies and simulation of trait evolution along them.

A tree is two parallel arrays: parent[v] (-1 for the root) and length[v], the
length of the branch above v. Every parent is numbered before its children.
Simulated trees with known rates are how we check that the estimators recover
the truth before using them on real phylogenies.
"""

import numpy as np


def yule_tree(n_tips, birth=1.0, rng=None):
    """Pure-birth (Yule) tree with n_tips tips."""
    rng = np.random.default_rng(rng)
    parent, start, end = [-1], [0.0], [np.nan]
    alive, t = [0], 0.0
    while len(alive) < n_tips:
        t += rng.exponential(1 / (birth * len(alive)))
        j = rng.integers(len(alive))
        v = alive[j]
        alive[j] = alive[-1]
        alive.pop()
        end[v] = t
        for _ in range(2):
            parent.append(v)
            start.append(t)
            end.append(np.nan)
            alive.append(len(parent) - 1)
    t += rng.exponential(1 / (birth * len(alive)))
    end = np.array(end)
    end[alive] = t
    return np.array(parent), end - np.array(start)


def tips(parent):
    """Indices of the tips (nodes that are nobody's parent)."""
    parent = np.asarray(parent)
    is_parent = np.zeros(parent.size, bool)
    is_parent[parent[parent >= 0]] = True
    return np.flatnonzero(~is_parent)


def simulate(model, parent, length, root_state=0, rng=None):
    """Simulate a walks.TraitModel along every branch (Gillespie); return the state at each node.

    root_state is the state at the top of the root's branch.
    """
    rng = np.random.default_rng(rng)
    k = model.k
    shifts = np.arange(k)
    state = np.empty(len(parent), dtype=np.int64)
    for v in range(len(parent)):
        x = int(root_state if parent[v] < 0 else state[parent[v]])
        t = 0.0
        while True:
            rates = model.flip_rates((x >> shifts) & 1)[0]
            total = rates.sum()
            if total == 0:
                break
            t += rng.exponential(1 / total)
            if t > length[v]:
                break
            x ^= 1 << int(rng.choice(k, p=rates / total))
        state[v] = x
    return state


def count_origins(state, parent, target):
    """Branches whose end is in target state but whose start is not: observed independent origins."""
    state, parent = np.asarray(state), np.asarray(parent)
    child = np.flatnonzero(parent >= 0)
    return int(((state[child] == target) & (state[parent[child]] != target)).sum())
