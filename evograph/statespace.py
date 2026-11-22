"""State graphs and the random-graph tools from "Random Graphs and Expansion".

A body plan with k binary traits is a state, stored as an integer from 0 to
2**k - 1 whose bit i is trait i. In the hypercube H_k two states are adjacent
when they differ in one trait. H_k has n = 2**k nodes, degree k = log2(n),
diameter k, and expansion exactly 1 (Section 5 of the notes).

Developmental constraints keep each hypercube edge with probability p, which
makes the state graph a random subgraph of H_k. G(n, p) and the configuration
model (Sections 1 and 2) serve as null models.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path


def states(k):
    """Bit matrix of all 2**k states; row x holds the k traits of state x."""
    x = np.arange(2**k)
    return ((x[:, None] >> np.arange(k)) & 1).astype(np.int8)


def hypercube(k, keep=None):
    """Adjacency matrix of H_k, keeping only the edges marked in keep (see constraint_mask)."""
    n = 2**k
    x = np.repeat(np.arange(n), k)
    y = x ^ (1 << np.tile(np.arange(k), n))
    m = np.ones(n * k, bool) if keep is None else np.asarray(keep, bool).reshape(-1)
    return sp.csr_matrix((np.ones(m.sum()), (x[m], y[m])), shape=(n, n))


def constraint_mask(k, p, rng=None):
    """Keep each edge of H_k independently with probability p.

    keep[x, i] says whether the edge between x and x with trait i flipped
    survives; both endpoints of an edge get the same answer.
    """
    rng = np.random.default_rng(rng)
    n = 2**k
    keep = np.zeros((n, k), bool)
    for i in range(k):
        low = np.flatnonzero(((np.arange(n) >> i) & 1) == 0)
        draw = rng.random(low.size) < p
        keep[low, i] = draw
        keep[low | (1 << i), i] = draw
    return keep


def gnp(n, p, rng=None):
    """G(n, p): each pair of the n nodes is joined independently with probability p."""
    rng = np.random.default_rng(rng)
    i, j = np.triu_indices(n, 1)
    m = rng.random(i.size) < p
    A = sp.csr_matrix((np.ones(m.sum()), (i[m], j[m])), shape=(n, n))
    return A + A.T


def configuration_model(degrees, rng=None):
    """Random multigraph with the given degrees, from a random pairing of half-edges.

    Self-loops and parallel edges are kept, as in the notes. A self-loop adds 2
    to its node's row, so the row sums equal the degrees.
    """
    rng = np.random.default_rng(rng)
    degrees = np.asarray(degrees)
    if degrees.sum() % 2:
        raise ValueError("degrees must sum to an even number")
    stubs = rng.permutation(np.repeat(np.arange(degrees.size), degrees))
    u, v = stubs[0::2], stubs[1::2]
    n = degrees.size
    A = sp.coo_matrix((np.ones(u.size), (u, v)), shape=(n, n)).tocsr()
    return A + A.T


def components(adj):
    """Connected-component label of every node."""
    return connected_components(adj, directed=False)[1]


def giant_fraction(adj):
    """Fraction of nodes in the largest connected component."""
    return np.bincount(components(adj)).max() / adj.shape[0]


def distances(adj, sources=None):
    """BFS distances (numbers of edges) from each source; inf where unreachable."""
    return shortest_path(adj, unweighted=True, directed=False, indices=sources)


def diameter(adj):
    """Largest finite distance between two nodes (exact; fine up to a few thousand nodes)."""
    d = distances(adj)
    return int(d[np.isfinite(d)].max())


def boundary(adj, S):
    """|e_out(S)|: total weight of edges with exactly one end in S."""
    x = np.zeros(adj.shape[0])
    x[np.asarray(S)] = 1
    return float(x @ (adj @ (1 - x)))


def surface_to_volume(adj, S):
    """sigma(S) = |e_out(S)| / min(|S|, n - |S|)."""
    x = np.zeros(adj.shape[0], bool)
    x[np.asarray(S)] = True
    size = x.sum()
    return boundary(adj, x) / min(size, x.size - size)


def expansion(adj, max_nodes=20):
    """Exact expansion alpha(G) = min sigma(S) over sets with |S| <= n/2, by brute force.

    The search is exponential in n, so it is only for small graphs; use
    spectral.expansion_bounds for anything larger.
    """
    A = adj.toarray() if sp.issparse(adj) else np.asarray(adj, float)
    n = A.shape[0]
    if n > max_nodes:
        raise ValueError(f"exact expansion is exponential in n; n = {n} > {max_nodes}")
    bits = 1 << np.arange(n)
    best = np.inf
    chunk = 1 << 16
    for start in range(1, 2**n, chunk):
        masks = np.arange(start, min(start + chunk, 2**n))
        X = ((masks[:, None] & bits) > 0).astype(float)
        size = X.sum(axis=1)
        ok = size <= n / 2
        if ok.any():
            cut = ((X[ok] @ A) * (1 - X[ok])).sum(axis=1)
            best = min(best, (cut / size[ok]).min())
    return best


def short_path_bound(n, d, alpha):
    """Expansion implies short paths: every pair is within 2 (d / alpha) log2(n) steps (Section 4)."""
    return 2 * (d / alpha) * np.log2(n)
