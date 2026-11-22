import numpy as np
import pytest

from evograph import trees, walks


def test_yule_tree_shape():
    parent, length = trees.yule_tree(50, rng=0)
    assert len(parent) == 99 and len(trees.tips(parent)) == 50
    assert (length > 0).all()
    assert all(parent[v] < v for v in range(1, len(parent)))


def test_long_branch_reaches_stationary():
    model = walks.TraitModel([1.0, 2.0], [2.0, 1.0])
    sigma = walks.stationary(model.transition_matrix())
    parent, length = np.array([-1]), np.array([20.0])
    rng = np.random.default_rng(0)
    ends = [trees.simulate(model, parent, length, rng=rng)[0] for _ in range(2000)]
    freq = np.bincount(ends, minlength=4) / len(ends)
    assert np.abs(freq - sigma).max() < 0.04


def test_count_origins():
    parent = np.array([-1, 0, 0, 1, 1])
    state = np.array([0, 3, 0, 3, 0])
    assert trees.count_origins(state, parent, target=3) == 1
