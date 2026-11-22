"""Random walks and Markov chains on state graphs ("Random Walks" notes).

Lazy walk. On a d-regular graph the lazy walk M' = I/2 + A/(2d) stays put with
probability 1/2, so it converges from any start (Theorem 5.1). Its second
eigenvalue is 1 - lambda_2 / (2d), and

    k = (2d / lambda_2) (1.5 ln n + ln(1/delta))

steps bring every state within relative error delta of the stationary
distribution (Section 5.3).

Evolutionary walk. Lineages follow a biased Markov chain on the same state
graph. Trait i is gained at rate r_gain[i] * exp(gamma * s_i(x)) and lost at
rate r_loss[i], where s_i(x) counts the present traits j with D[j, i] = 1
(trait j helps trait i). Edges removed by constraints get rate 0. Rates become
a lazy discrete-time chain by uniformization,

    P = I + Q / (2 * maximum exit rate),

so every state keeps probability at least 1/2 of staying put, as in M'. With
equal rates and no constraints, P is exactly the lazy walk on H_k. By Theorem
6.3, an irreducible chain has a unique stationary distribution sigma, and the
expected return time to state x is 1 / sigma(x).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .statespace import states


def lazy_walk(adj):
    """M' = I/2 + D^{-1} A / 2, which is I/2 + A/(2d) on a d-regular graph. Isolated nodes stay put."""
    A = sp.csr_matrix(adj, dtype=float)
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    stuck = sp.diags((deg == 0).astype(float))
    return (sp.identity(A.shape[0]) + sp.diags(inv) @ A + stuck) / 2


def mixing_steps(n, d, lam2, delta=0.01):
    """Steps after which the lazy walk is within relative error delta of stationary (Section 5.3)."""
    return 2 * d / lam2 * (1.5 * np.log(n) + np.log(1 / delta))


def max_relative_error(P, start, pi, steps):
    """Max over states of |pi - p_t| / pi after each of the first `steps` steps from `start`."""
    PT = sp.csr_matrix(P).T.tocsr()
    p = np.zeros(PT.shape[0])
    p[start] = 1
    errs = np.empty(steps)
    for t in range(steps):
        p = PT @ p
        errs[t] = np.max(np.abs(pi - p) / pi)
    return errs


class TraitModel:
    """Biased walk on the trait hypercube.

    r_gain, r_loss: length-k arrays of baseline gain and loss rates.
    gamma: coupling strength; each present helper trait multiplies a gain rate by e^gamma.
    D: k x k 0/1 matrix, D[j, i] = 1 when trait j helps trait i (the trait-dependency graph).
    keep: optional (2**k, k) edge mask from statespace.constraint_mask.
    """

    def __init__(self, r_gain, r_loss, gamma=0.0, D=None, keep=None):
        self.r_gain = np.asarray(r_gain, float)
        self.r_loss = np.asarray(r_loss, float)
        self.k = self.r_gain.size
        self.gamma = float(gamma)
        self.D = np.zeros((self.k, self.k)) if D is None else np.asarray(D, float)
        self.keep = None if keep is None else np.asarray(keep, bool)

    def flip_rates(self, X):
        """Rate of flipping each trait, for each state in X (rows are bit vectors)."""
        X = np.atleast_2d(X)
        gain = self.r_gain * np.exp(self.gamma * (X @ self.D))
        rates = np.where(X == 0, gain, self.r_loss)
        if self.keep is not None:
            rates = rates * self.keep[X @ (1 << np.arange(self.k))]
        return rates

    def generator(self):
        """Continuous-time generator Q on all 2**k states."""
        k, n = self.k, 2**self.k
        R = self.flip_rates(states(k)).reshape(-1)
        x = np.repeat(np.arange(n), k)
        y = x ^ (1 << np.tile(np.arange(k), n))
        Q = sp.csr_matrix((R, (x, y)), shape=(n, n))
        return (Q - sp.diags(np.asarray(Q.sum(axis=1)).ravel())).tocsr()

    def steps_per_unit_time(self):
        """Uniformization rate: one unit of model time is this many steps of P."""
        return 2 * (-self.generator().diagonal()).max()

    def transition_matrix(self):
        """Lazy discrete-time chain P = I + Q / (2 * maximum exit rate)."""
        Q = self.generator()
        return (sp.identity(Q.shape[0]) + Q / self.steps_per_unit_time()).tocsr()


def stationary(P):
    """Stationary distribution: sigma P = sigma, entries summing to 1. Needs an irreducible chain."""
    n = P.shape[0]
    A = (sp.csr_matrix(P).T - sp.identity(n)).tolil()
    A[n - 1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    return spla.spsolve(A.tocsc(), b)


def return_times(sigma):
    """Expected steps to return to each state, 1 / sigma (Theorem 6.3)."""
    return 1 / np.asarray(sigma)


def hitting_times(P, targets):
    """Expected steps to first reach any target state, from every state."""
    n = P.shape[0]
    T = np.zeros(n, bool)
    T[np.asarray(targets)] = True
    S = ~T
    P = sp.csr_matrix(P)
    A = sp.identity(S.sum()) - P[S][:, S]
    h = np.zeros(n)
    h[S] = np.atleast_1d(spla.spsolve(A.tocsc(), np.ones(S.sum())))
    return h


def flow_graph(P, sigma):
    """Undirected weights w(x, y) = (sigma(x) P(x, y) + sigma(y) P(y, x)) / 2.

    Passing these weights and sigma to spectral.fiedler or spectral.sweep_cut
    gives the chain's conductance; the cut weight equals the stationary flow
    across the cut even when the chain is not reversible.
    """
    Pi = sp.diags(sigma)
    P = sp.csr_matrix(P)
    W = (Pi @ P + P.T @ Pi) / 2
    return (W - sp.diags(W.diagonal())).tocsr()


def independent_prediction(r_gain, r_loss):
    """Stationary probability of having every trait if traits evolved independently."""
    a, b = np.asarray(r_gain, float), np.asarray(r_loss, float)
    return float(np.prod(a / (a + b)))


def integration_index(model):
    """Observed stationary mass on the all-traits state over the independent-traits prediction.

    1 means the traits evolve independently; above 1 means linked traits pull
    lineages toward the full body plan. Exact only without edge constraints.
    """
    sigma = stationary(model.transition_matrix())
    return sigma[-1] / independent_prediction(model.r_gain, model.r_loss)
