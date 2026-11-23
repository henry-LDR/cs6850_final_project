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


def read_newick(text):
    """Parse a Newick string into (parent, length, names), numbering parents before children.

    Internal-node labels (such as bootstrap values) are kept in names; nodes
    without a label get "".
    """
    text = text.strip().rstrip(";").strip()
    parent, length, names = [], [], []
    stack, cur, i = [], None, 0

    def new(p):
        parent.append(p)
        length.append(0.0)
        names.append("")
        return len(parent) - 1

    while i < len(text):
        c = text[i]
        if c == "(":
            stack.append(new(stack[-1] if stack else -1))
            cur = None
            i += 1
        elif c == ",":
            cur = None
            i += 1
        elif c == ")":
            cur = stack.pop()
            i += 1
        elif c.isspace():
            i += 1
        else:
            if cur is None:
                cur = new(stack[-1] if stack else -1)
            if c == ":":
                j = i + 1
                while j < len(text) and text[j] not in ",()":
                    j += 1
                length[cur] = float(text[i + 1 : j])
            else:
                j = i
                while j < len(text) and text[j] not in ",():":
                    j += 1
                names[cur] = text[i:j].strip().strip("'\"")
            i = j
    return np.array(parent), np.array(length), names


def read_nexus(text):
    """First tree in a NEXUS file, applying its TRANSLATE table if it has one."""
    import re

    body = text[text.lower().index("begin trees") :]
    names = {}
    m = re.search(r"translate(.*?);", body, re.S | re.I)
    if m:
        for entry in m.group(1).split(","):
            parts = entry.strip().split(None, 1)
            if len(parts) == 2:
                names[parts[0]] = parts[1].strip().strip("'\"")
    tree = re.search(r"\btree\s+[^=]+=\s*(?:\[[^\]]*\]\s*)?(.*?;)", body, re.S | re.I).group(1)
    tree = re.sub(r"\[[^\]]*\]", "", tree)  # drop comments such as [&R]
    parent, length, labels = read_newick(tree)
    return parent, length, [names.get(x, x) for x in labels]


def subtree(parent, length, names, root):
    """The clade below node root, reindexed so parents still come before children."""
    parent = np.asarray(parent)
    keep = np.zeros(len(parent), bool)
    keep[root] = True
    for v in range(root + 1, len(parent)):
        if parent[v] >= 0 and keep[parent[v]]:
            keep[v] = True
    old = np.flatnonzero(keep)
    new = {o: i for i, o in enumerate(old)}
    par = np.array([-1 if o == root else new[parent[o]] for o in old])
    return par, np.where(par < 0, 0.0, np.asarray(length)[old]), [names[o] for o in old]


def mrca(parent, nodes):
    """Most recent common ancestor of the given nodes."""
    def path(v):
        out = []
        while v >= 0:
            out.append(v)
            v = parent[v]
        return out
    common = set(path(nodes[0]))
    for v in nodes[1:]:
        common &= set(path(v))
    return max(common)  # deepest shared node has the largest index


def rescale(length, parent, height=1.0):
    """Scale branch lengths so the deepest tip sits at the given height above the root."""
    depth = np.zeros(len(parent))
    for v in range(1, len(parent)):
        depth[v] = depth[parent[v]] + length[v]
    return length * (height / depth.max())


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
