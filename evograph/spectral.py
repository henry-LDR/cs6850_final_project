"""Spectral analysis of state graphs ("Spectral Analysis of Graphs").

For an undirected graph with adjacency A and degrees D, the Laplacian is
L = D - A, with x^T L x equal to the sum over edges of (x_i - x_j)^2 and
eigenvalues 0 = lambda_1 <= lambda_2 <= ... The Fiedler vector (the
eigenvector of lambda_2) is the best one-dimensional embedding of the graph,
and Claim 2.2 ties lambda_2 to the expansion alpha:

    2 alpha >= lambda_2 >= alpha^2 / (2 Delta),   Delta = maximum degree.

Every function also takes an optional node measure m, such as the stationary
distribution of a biased walk. Then |S| becomes m(S), the eigenproblem becomes
L v = lambda M v with M = diag(m), and the surface-to-volume ratio becomes the
conductance of S.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _no_loops(adj):
    A = sp.csr_matrix(adj, dtype=float)
    return (A - sp.diags(A.diagonal())).tocsr()


def laplacian(adj):
    """L = D - A. Edge weights are allowed; self-loops are ignored."""
    A = _no_loops(adj)
    return sp.diags(np.asarray(A.sum(axis=1)).ravel()) - A


def fiedler(adj, measure=None, dense_limit=4096):
    """Return (lambda_2, Fiedler vector) for L v = lambda M v."""
    L = laplacian(adj)
    n = L.shape[0]
    m = np.ones(n) if measure is None else np.asarray(measure, float)
    s = 1 / np.sqrt(m)
    N = sp.diags(s) @ L @ sp.diags(s)
    if n <= dense_limit:
        w, U = np.linalg.eigh(N.toarray())
    else:
        w, U = spla.eigsh(N.tocsc(), k=2, sigma=-1e-6, which="LM")
        order = np.argsort(w)
        w, U = w[order], U[:, order]
    return w[1], s * U[:, 1]


def sweep_cut(adj, vector=None, measure=None):
    """Best prefix set of the sorted Fiedler vector.

    Returns (S, ratio), where ratio = w(e_out(S)) / min(m(S), m(V - S)) is the
    smallest over all prefix sets. The ratio is an upper bound on the
    expansion (or conductance), and S is the bottleneck spectral partitioning
    finds.
    """
    A = _no_loops(adj)
    n = A.shape[0]
    m = np.ones(n) if measure is None else np.asarray(measure, float)
    v = fiedler(adj, measure)[1] if vector is None else np.asarray(vector)
    order = np.argsort(v)
    pos = np.empty(n, int)
    pos[order] = np.arange(n)
    deg = np.asarray(A.sum(axis=1)).ravel()
    total = m.sum()
    cut = mass = 0.0
    best, best_t = np.inf, 1
    for t, u in enumerate(order[:-1]):
        lo, hi = A.indptr[u], A.indptr[u + 1]
        nbrs, w = A.indices[lo:hi], A.data[lo:hi]
        cut += deg[u] - 2 * w[pos[nbrs] < t].sum()
        mass += m[u]
        ratio = cut / min(mass, total - mass)
        if ratio < best:
            best, best_t = ratio, t + 1
    return order[:best_t], best


def expansion_bounds(adj):
    """Bounds lambda_2 / 2 <= alpha <= upper, from Claim 2.2 and the sweep cut."""
    lam2, v = fiedler(adj)
    Delta = np.asarray(_no_loops(adj).sum(axis=1)).max()
    _, sweep = sweep_cut(adj, v)
    return lam2 / 2, min(np.sqrt(2 * Delta * lam2), sweep)
