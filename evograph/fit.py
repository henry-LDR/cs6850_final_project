"""Maximum-likelihood fitting of a TraitModel to traits observed at the tips of a tree.

The likelihood comes from Felsenstein's pruning algorithm. Moving a
partial-likelihood vector up a branch of length t multiplies it by exp(Q t).
Large state spaces compute this by uniformization, which runs the lazy walk
P = I + Q / Lambda for a Poisson(Lambda t) number of steps:

    exp(Q t) L = sum_n Poisson(n; Lambda t) P^n L,

the same construction that turns the trait walk into a discrete chain in
walks.TraitModel. It needs only sparse products with P and every term is
nonnegative. Small state spaces (at most DENSE_LIMIT states) compute every
branch's exp(Q t) in one batched expm call instead.

Tips can carry a misclassification rate: each observed trait is wrong with
probability error, independently, as for an unreliable phenotypic test.
"""

import numpy as np
import scipy.linalg as sla
import scipy.optimize as opt
import scipy.sparse as sp

from .walks import TraitModel

DENSE_LIMIT = 64
MAX_STEPS = 1e6  # uniformization budget per branch; beyond it the likelihood is reported as -inf


class _TooStiff(Exception):
    pass


def _propagator(Q, lengths, dense_limit):
    """Return f(v, t, L) = exp(Q t) L for the branch above node v, of length t.

    Small state spaces precompute every branch's exp(Q t) in one batched scipy
    expm call. (Diagonalizing Q is faster still but its eigenvectors can be
    ill-conditioned, which makes the likelihood too noisy for finite-difference
    gradients.)
    """
    if Q.shape[0] > dense_limit:
        return _uniformization(Q)
    nodes = [v for v, t in lengths.items() if t > 0]
    A = Q.toarray()
    mats = dict(zip(nodes, sla.expm(A[None] * np.array([lengths[v] for v in nodes])[:, None, None]))) if nodes else {}
    return lambda v, t, L: mats[v] @ L


def _uniformization(Q, chunk=30.0):
    """Return f(v, t, L) = exp(Q t) L computed by uniformization (see the module docstring)."""
    lam = max(float(-Q.diagonal().min()), 1e-300)
    P = (sp.identity(Q.shape[0], format="csr") + Q / lam).tocsr()

    def f(v, t, L):
        mu_total = lam * t
        if mu_total > MAX_STEPS:
            raise _TooStiff
        pieces = max(1, int(np.ceil(mu_total / chunk)))  # exp(Qt) = exp(Qt/c)^c keeps Poisson means small
        mu = mu_total / pieces
        for _ in range(pieces):
            w = np.exp(-mu)
            out, term, n, mass = w * L, L, 0, w
            while mass < 1 - 1e-15 and n < 1000:
                n += 1
                term = P @ term
                w *= mu / n
                out = out + w * term
                mass += w
            L = out
        return L

    return f


def _mismatches(observation, k):
    bits = (np.arange(2**k)[:, None] >> np.arange(k)) & 1
    obs = np.asarray([int(c) for c in observation])
    known = obs != 2
    return (bits[:, known] != obs[known]).sum(axis=1), int(known.sum())


def tip_vector(observation, k, error=0.0):
    """Partial likelihood for one tip from a sequence of 0, 1 and 2 (unknown), one entry per trait.

    With error > 0 each known trait is misrecorded with that probability, so the
    tip vector is error**m * (1 - error)**(known - m) for a state m traits away."""
    m, known = _mismatches(observation, k)
    return error**m * (1 - error) ** (known - m)


def log_likelihood(model, parent, length, tip_vectors, root_state=0, dense_limit=DENSE_LIMIT):
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
    propagate = _propagator(Q, {v: length[v] for v in range(n) if informative[v] and parent[v] >= 0}, dense_limit)
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
        try:
            msg = propagate(v, length[v], L) if length[v] > 0 else L
        except _TooStiff:  # rates so fast that a branch needs over MAX_STEPS steps: hopeless parameters
            return -np.inf
        acc[parent[v]] *= np.maximum(msg, 0.0)
    raise ValueError("tree has no root")


def fit_trait_model(k, parent, length, tip_vectors=None, D=None, root_state=0, fit_gamma=True, x0=None,
                    observations=None, fit_error=False, pairwise=False, keep=None):
    """Fit per-trait gain rates, one shared loss rate, and optionally the coupling and a
    tip misclassification rate.

    Coupling is either one gamma on the dependency graph D (fit_gamma; D defaults to
    all-to-all) or, with pairwise=True (or a k x k boolean mask of the pairs to fit, the rest
    fixed at 0), a separate weight Gamma[j, i] for every ordered pair:
    holding trait j multiplies trait i's gain rate by exp(Gamma[j, i]). The fitted Gamma is
    a directed, weighted trait-dependency graph. keep is an optional edge mask (see
    walks.TraitModel) for impossible states.

    Give the data either as tip_vectors (node -> partial likelihood) or as observations
    (node -> sequence of 0, 1 and 2); fit_error needs observations. Parameters are
    [log gains (k), log loss, gamma or the free Gamma entries (row-major), logit error].
    Returns a dict with the fitted TraitModel, parameters and maximized log-likelihood.
    """
    D = np.ones((k, k)) - np.eye(k) if D is None else np.asarray(D, float)
    if fit_error and observations is None:
        raise ValueError("fit_error needs observations")
    if pairwise is not False and pairwise is not None:
        fit_gamma = False
        off = ~np.eye(k, dtype=bool) if pairwise is True else (np.asarray(pairwise, bool) & ~np.eye(k, dtype=bool))
        pairwise = True
    else:
        pairwise, off = False, np.zeros((k, k), bool)
    n_pair = int(off.sum())
    if observations is not None:
        mism = {v: _mismatches(o, k) for v, o in observations.items()}
    i_gamma = k + 1 if fit_gamma else None
    i_error = k + 1 + int(fit_gamma) + n_pair if fit_error else None
    n_par = k + 1 + int(fit_gamma) + n_pair + int(fit_error)

    def build(theta):
        if pairwise:
            G = np.zeros((k, k))
            G[off] = theta[k + 1 : k + 1 + n_pair]
            return TraitModel(np.exp(theta[:k]), np.full(k, np.exp(theta[k])), gamma=1.0, D=G, keep=keep)
        gamma = theta[i_gamma] if fit_gamma else 0.0
        return TraitModel(np.exp(theta[:k]), np.full(k, np.exp(theta[k])), gamma=gamma, D=D, keep=keep)

    def error_of(theta):
        return 1 / (1 + np.exp(-theta[i_error])) if fit_error else 0.0

    def tips(theta):
        if observations is None:
            return tip_vectors
        e = error_of(theta)
        return {v: e**m * (1 - e) ** (n - m) for v, (m, n) in mism.items()}

    def nll(theta):
        ll = log_likelihood(build(theta), parent, length, tips(theta), root_state)
        return 1e12 if not np.isfinite(ll) else -ll

    if x0 is None:
        x0 = np.r_[np.zeros(k), np.log(0.1), [0.0] * (int(fit_gamma) + n_pair), [np.log(0.02 / 0.98)] * int(fit_error)]
    bounds = [(-8, 6)] * (k + 1) + [(-4, 4)] * int(fit_gamma) + [(-6, 6)] * n_pair + [(-11.5, -0.85)] * int(fit_error)
    res = opt.minimize(nll, np.asarray(x0, float), method="L-BFGS-B", bounds=bounds)
    model = build(res.x)
    return {
        "model": model,
        "gain": model.r_gain,
        "loss": float(model.r_loss[0]),
        "gamma": model.gamma if not pairwise else None,
        "Gamma": model.D.tolist() if pairwise else None,
        "error": float(error_of(res.x)),
        "theta": res.x,
        "loglik": -float(res.fun),
        "n_params": n_par,
        "converged": bool(res.success),
    }
