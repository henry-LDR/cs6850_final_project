import numpy as np
import pytest

from evograph import fit, trees, walks


def test_newick_parser():
    parent, length, names = trees.read_newick("((A:1,B:2)90:0.5,C:3);")
    assert len(parent) == 5 and parent[0] == -1
    tip_names = sorted(names[v] for v in trees.tips(parent))
    assert tip_names == ["A", "B", "C"]
    assert length[names.index("B")] == 2.0 and names[1] == "90"


def test_single_branch_matches_two_state_formula():
    # Root in state 0, one tip at distance t observed in state 1: P01(t) = a/(a+b) (1 - e^{-(a+b)t}).
    a, b, t = 0.7, 0.3, 1.5
    parent, length = np.array([-1, 0]), np.array([0.0, t])
    model = walks.TraitModel([a], [b])
    ll = fit.log_likelihood(model, parent, length, {1: fit.tip_vector([1], 1)})
    assert ll == pytest.approx(np.log(a / (a + b) * (1 - np.exp(-(a + b) * t))))


def test_unknown_traits_and_missing_tips_do_not_change_the_likelihood():
    parent, length = trees.yule_tree(20, rng=1)
    length = trees.rescale(length, parent)
    model = walks.TraitModel([1.0, 0.5], [0.2, 0.2], gamma=0.3, D=[[0, 1], [1, 0]])
    tips = trees.tips(parent)
    data = {v: fit.tip_vector([1, 0], 2) for v in tips[:10]}
    ll = fit.log_likelihood(model, parent, length, data)
    with_unknowns = {**data, **{v: fit.tip_vector([2, 2], 2) for v in tips[10:]}}
    assert fit.log_likelihood(model, parent, length, with_unknowns) == pytest.approx(ll)


def test_fit_recovers_rates_from_a_simulated_tree():
    k = 2
    truth = walks.TraitModel([1.5, 0.5], [0.4, 0.4])
    parent, length = trees.yule_tree(300, rng=3)
    length = trees.rescale(length, parent, height=3.0)
    state = trees.simulate(truth, parent, length, rng=4)
    data = {v: fit.tip_vector([(state[v] >> i) & 1 for i in range(k)], k) for v in trees.tips(parent)}
    result = fit.fit_trait_model(k, parent, length, data, fit_gamma=False)
    assert result["gain"] == pytest.approx(truth.r_gain, rel=0.5)
    assert result["loss"] == pytest.approx(0.4, rel=0.6)


def test_dense_and_sparse_paths_agree():
    rng = np.random.default_rng(3)
    k = 3
    model = walks.TraitModel([0.6, 0.4, 0.3], [0.2] * 3, gamma=0.7, D=np.ones((k, k)) - np.eye(k))
    parent, length = trees.yule_tree(40, rng=rng)
    states = trees.simulate(model, parent, length, 0, rng=rng)
    data = {v: fit.tip_vector([(states[v] >> i) & 1 for i in range(k)], k) for v in trees.tips(parent)}
    dense = fit.log_likelihood(model, parent, length, data)
    sparse = fit.log_likelihood(model, parent, length, data, dense_limit=0)
    assert np.isclose(dense, sparse, rtol=1e-6)


def test_tip_vector_error():
    exact = fit.tip_vector([1, 0, 2], 3)
    assert np.allclose(exact, fit.tip_vector([1, 0, 2], 3, error=0.0))
    noisy = fit.tip_vector([1, 0, 2], 3, error=0.1)
    # state 0b001 matches both known traits; 0b000 misses one; 0b010 misses both
    assert np.isclose(noisy[0b001], 0.81) and np.isclose(noisy[0b000], 0.09) and np.isclose(noisy[0b010], 0.01)


def test_uniformization_matches_expm():
    import scipy.linalg as sla
    k = 5
    model = walks.TraitModel(np.linspace(0.5, 3, k), [0.4] * k, gamma=0.6, D=np.ones((k, k)) - np.eye(k))
    Q = model.generator()
    L = np.random.default_rng(0).random(2**k)
    f = fit._uniformization(Q.tocsr())
    for t in (0.01, 1.0, 20.0):
        assert np.allclose(f(0, t, L), sla.expm(Q.toarray() * t) @ L, rtol=1e-10)
