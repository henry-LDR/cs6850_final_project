"""Cascades on networks (Kleinberg, "Cascading Behavior in Networks", AGT chapter 24).

All three processes are progressive: an adopter never switches back.

  threshold_cascade    one threshold q for every node, the coordination-game rule (Section 2)
  linear_threshold     weighted influence, thresholds theta_v uniform on [0, 1] (Section 3)
  independent_cascade  each new adopter u gets one chance to convert v, with probability P[u, v]

greedy_seeds is the hill-climbing algorithm for finding influential seed sets.
When the spread function is monotone and submodular, as in the Linear
Threshold and Independent Cascade models, greedy is within a factor 1 - 1/e of
the best set of the same size (Theorems 4.1 to 4.3). Critical-mass rules
("adopt once k neighbors have") are not submodular, so that guarantee fails
for them, which matters for developmental thresholds.

Graphs are weight matrices W with W[u, v] the influence of u on v. In this
project the nodes are traits (the trait-dependency graph D) or species (a
gene-transfer network for bacteria).
"""

import numpy as np
import scipy.sparse as sp


def _csr(W):
    return sp.csr_matrix(W, dtype=float)


def _seed_mask(n, seeds):
    active = np.zeros(n, bool)
    active[np.asarray(seeds, int)] = True
    return active


def threshold_cascade(W, seeds, q):
    """Progressive cascade where v adopts once a share q of its incoming influence has adopted.

    Returns (adopted, rounds).
    """
    W = _csr(W)
    n = W.shape[0]
    WT = W.T.tocsr()
    indeg = np.asarray(W.sum(axis=0)).ravel()
    active = _seed_mask(n, seeds)
    rounds = 0
    while True:
        share = np.divide(WT @ active.astype(float), indeg, out=np.zeros(n), where=indeg > 0)
        new = ~active & (indeg > 0) & (share >= q - 1e-12)
        if not new.any():
            return active, rounds
        active |= new
        rounds += 1


def linear_threshold(W, seeds, rng=None):
    """Linear Threshold Model. Incoming weights of each node must sum to at most 1."""
    rng = np.random.default_rng(rng)
    W = _csr(W)
    n = W.shape[0]
    if (np.asarray(W.sum(axis=0)).ravel() > 1 + 1e-9).any():
        raise ValueError("incoming weights must sum to at most 1")
    WT = W.T.tocsr()
    theta = rng.random(n)
    active = _seed_mask(n, seeds)
    while True:
        new = ~active & (WT @ active.astype(float) >= theta)
        if not new.any():
            return active
        active |= new


def independent_cascade(P, seeds, rng=None):
    """Independent Cascade Model: each new adopter u converts v with probability P[u, v], once."""
    rng = np.random.default_rng(rng)
    P = _csr(P)
    active = _seed_mask(P.shape[0], seeds)
    frontier = np.flatnonzero(active)
    while frontier.size:
        hits = []
        for u in frontier:
            lo, hi = P.indptr[u], P.indptr[u + 1]
            nbrs = P.indices[lo:hi]
            hit = nbrs[(rng.random(hi - lo) < P.data[lo:hi]) & ~active[nbrs]]
            active[hit] = True
            hits.append(hit)
        frontier = np.unique(np.concatenate(hits))
    return active


def expected_spread(process, seeds, runs=200, rng=None):
    """Monte Carlo estimate of f(S), the expected number of adopters.

    process(seeds, rng) must return a boolean adopter mask. Fix rng (an int)
    when comparing seed sets, so every set sees the same random draws.
    """
    rng = np.random.default_rng(rng)
    return float(np.mean([process(seeds, rng).sum() for _ in range(runs)]))


def greedy_seeds(f, candidates, budget):
    """Hill-climbing: repeatedly add the candidate with the largest value of f(chosen + [c])."""
    chosen = []
    for _ in range(budget):
        rest = [c for c in candidates if c not in chosen]
        chosen.append(max(rest, key=lambda c: f(chosen + [c])))
    return chosen
