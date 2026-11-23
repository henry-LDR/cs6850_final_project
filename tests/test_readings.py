"""Checks that the code reproduces results stated in the course readings."""

from math import comb

import numpy as np
import pytest

from evograph import spectral, statespace, walks


# Random Graphs and Expansion --------------------------------------------------

def test_hypercube_has_expansion_one_and_diameter_k():
    A = statespace.hypercube(4)
    assert statespace.expansion(A) == pytest.approx(1.0)
    assert statespace.diameter(A) == 4


def test_expansion_implies_short_paths():
    for k in (3, 5, 7):
        A = statespace.hypercube(k)
        assert statespace.diameter(A) <= statespace.short_path_bound(2**k, k, 1.0)


def test_configuration_model_has_requested_degrees():
    degrees = np.full(50, 3)
    A = statespace.configuration_model(degrees, rng=0)
    assert np.array_equal(np.asarray(A.sum(axis=1)).ravel(), degrees)


def test_gnp_expected_degree():
    n, c = 400, 6
    A = statespace.gnp(n, c / (n - 1), rng=0)
    assert np.asarray(A.sum(axis=1)).mean() == pytest.approx(c, rel=0.1)


def test_constraints_are_symmetric_and_p1_keeps_everything():
    k = 6
    A = statespace.hypercube(k, statespace.constraint_mask(k, 0.5, rng=1))
    assert (A != A.T).nnz == 0
    full = statespace.hypercube(k, statespace.constraint_mask(k, 1.0, rng=1))
    assert statespace.giant_fraction(full) == 1.0


# Spectral Analysis of Graphs --------------------------------------------------

def test_hypercube_laplacian_spectrum():
    for k in (3, 4, 5):
        L = spectral.laplacian(statespace.hypercube(k)).toarray()
        values, counts = np.unique(np.round(np.linalg.eigvalsh(L)).astype(int), return_counts=True)
        assert list(values) == [2 * j for j in range(k + 1)]
        assert list(counts) == [comb(k, j) for j in range(k + 1)]


def test_claim_2_2_bounds_hold():
    for seed in range(5):
        A = statespace.gnp(14, 0.35, rng=seed)
        if statespace.giant_fraction(A) < 1:
            continue
        alpha = statespace.expansion(A)
        lo, hi = spectral.expansion_bounds(A)
        assert lo - 1e-9 <= alpha <= hi + 1e-9


def test_sweep_cut_on_the_hypercube():
    # lambda_2 = 2 has multiplicity k, so the Fiedler vector is not unique. Any
    # sweep is an upper bound on alpha = 1 and obeys Cheeger's sqrt(2 Delta lambda_2);
    # sweeping along a single trait coordinate recovers the half-cube cut exactly.
    k = 5
    A = statespace.hypercube(k)
    _, ratio = spectral.sweep_cut(A)
    assert 1.0 - 1e-9 <= ratio <= np.sqrt(2 * k * 2.0)
    coordinate = (np.arange(2**k) & 1) * 2.0 - 1
    S, ratio = spectral.sweep_cut(A, vector=coordinate)
    assert len(S) == 16 and ratio == pytest.approx(1.0)


# Random Walks -----------------------------------------------------------------

def test_unbiased_trait_model_is_the_lazy_walk():
    k = 5
    P = walks.TraitModel(np.ones(k), np.ones(k)).transition_matrix()
    M = walks.lazy_walk(statespace.hypercube(k))
    assert abs(P - M).max() < 1e-12


def test_lazy_walk_mixes_within_the_bound():
    k = 6
    n, delta = 2**k, 0.01
    M = walks.lazy_walk(statespace.hypercube(k))
    steps = int(np.ceil(walks.mixing_steps(n, k, lam2=2.0, delta=delta)))
    errs = walks.max_relative_error(M, 0, np.full(n, 1 / n), steps)
    assert errs[-1] <= delta


def test_return_time_is_one_over_sigma():
    model = walks.TraitModel([1.0, 2.0], [3.0, 1.0])
    P = model.transition_matrix()
    sigma = walks.stationary(P)
    # Expected return time to x = 1 + sum_y P(x, y) * (hitting time from y to x).
    x = 3
    h = walks.hitting_times(P, [x])
    assert 1 + P.toarray()[x] @ h == pytest.approx(walks.return_times(sigma)[x])


# Model-specific checks ----------------------------------------------------------

def test_independent_traits_factorize():
    model = walks.TraitModel([1.0, 2.0, 0.5], [1.0, 1.0, 2.0])
    sigma = walks.stationary(model.transition_matrix())
    assert sigma.sum() == pytest.approx(1.0)
    assert walks.integration_index(model) == pytest.approx(1.0)


def test_coupling_pulls_lineages_to_the_full_body_plan():
    D = np.ones((3, 3)) - np.eye(3)
    model = walks.TraitModel(np.ones(3), np.ones(3), gamma=1.0, D=D)
    assert walks.integration_index(model) > 1.5


def test_conductance_finds_a_planted_slow_trait():
    rates = np.array([0.01, 1.0, 1.0, 1.0])
    P = walks.TraitModel(rates, rates).transition_matrix()
    sigma = walks.stationary(P)
    W = walks.flow_graph(P, sigma)
    S, phi = spectral.sweep_cut(W, measure=sigma)
    trait0 = (np.asarray(S) & 1)
    assert len(S) == 8 and (trait0 == trait0[0]).all()


# Cascading Behavior in Networks -------------------------------------------------

def _cycle(n):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1
    return A
