"""Maximum-likelihood fitting of a TraitModel to traits observed at the tips of a tree.

The likelihood comes from Felsenstein's pruning algorithm. Moving a
partial-likelihood vector up a branch of length t multiplies it by exp(Q t),
which expm_multiply computes with sparse matrix-vector products, so 10 traits
(1,024 states) take about a second per evaluation.
"""

import numpy as np
import scipy.optimize as opt
import scipy.sparse.linalg as spla

from .walks import TraitModel


def tip_vector(observation, k):
    """Partial likelihood for one tip from a sequence of 0, 1 and 2 (unknown), one entry per trait."""
    bits = (np.arange(2**k)[:, None] >> np.arange(k)) & 1
    obs = np.asarray([int(c) for c in observation])
    known = obs != 2
    return (bits[:, known] == obs[known]).all(axis=1).astype(float)


def log_likelihood(model, parent, length, tip_vectors, root_state=0):
    """Log-likelihood of the tip data. tip_vectors maps node index to a partial likelihood;
    nodes absent from it carry no data. The root starts in root_state."""
    Q = model.generator().tocsc()
    n, N = len(parent), 2**model.k
    # Subtrees without data contribute exp(Q t) 1 = 1, so their branches are skipped.
    informative = np.zeros(n, bool)
    for v in range(n - 1, -1, -1):
        if v in tip_vectors:
            informative[v] = True
        if informative[v] and parent[v] >= 0:
            informative[parent[v]] = True
    acc = np.ones((n, N))
    logscale = 0.0
    for v in range(n - 1, -1, -1):
        if not informative[v] and parent[v] >= 0:
            continue
        L = acc[v] * tip_vectors[v] if v in tip_vectors else acc[v]
        s = L.max()
        if s <= 0:
            return -np.inf
        L = L / s
        logscale += np.log(s)
        if parent[v] < 0:
            return float(np.log(L[root_state]) + logscale) if L[root_state] > 0 else -np.inf
        msg = spla.expm_multiply(Q * length[v], L) if length[v] > 0 else L
        acc[parent[v]] *= np.maximum(msg, 0.0)
    raise ValueError("tree has no root")


def fit_trait_model(k, parent, length, tip_vectors, D=None, root_state=0, fit_gamma=True, x0=None):
    """Fit per-trait gain rates, one shared loss rate and (optionally) the coupling gamma.

    D defaults to all-to-all coupling (every present trait helps every absent one).
    Returns a dict with the fitted TraitModel, parameters and maximized log-likelihood.
    """
    D = np.ones((k, k)) - np.eye(k) if D is None else np.asarray(D, float)

    def build(theta):
        gamma = theta[k + 1] if fit_gamma else 0.0
        return TraitModel(np.exp(theta[:k]), np.full(k, np.exp(theta[k])), gamma=gamma, D=D)

    def nll(theta):
        ll = log_likelihood(build(theta), parent, length, tip_vectors, root_state)
        return 1e12 if not np.isfinite(ll) else -ll

    n_par = k + 2 if fit_gamma else k + 1
    x0 = np.r_[np.zeros(k), np.log(0.1), 0.0][:n_par] if x0 is None else np.asarray(x0)[:n_par]
    bounds = [(-8, 6)] * (k + 1) + [(-4, 4)] * (n_par - k - 1)
    res = opt.minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    model = build(res.x)
    return {
        "model": model,
        "gain": model.r_gain,
        "loss": float(model.r_loss[0]),
        "gamma": model.gamma,
        "loglik": -float(res.fun),
        "n_params": n_par,
        "converged": bool(res.success),
    }
