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
