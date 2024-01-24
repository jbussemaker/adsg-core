import pytest
import numpy as np
from adsg_core.optimization.assign_enc.matrix import *


@pytest.fixture
def nodes():
    return [Node([1]) for _ in range(10)]


@pytest.fixture
def inf_nodes():
    return [Node(min_conn=1) for _ in range(10)]


def test_conn_override_map(nodes, inf_nodes):
    assert AggregateAssignmentMatrixGenerator.create(src=[nodes[0]], tgt=[nodes[1]])._get_max_conn() == 1

    assert AggregateAssignmentMatrixGenerator.create(src=[nodes[0], inf_nodes[0]], tgt=[nodes[1]])._get_max_conn() == 2
    assert AggregateAssignmentMatrixGenerator.create(src=[nodes[0], inf_nodes[0]],
                                                     tgt=[nodes[1], inf_nodes[1]])._get_max_conn() == 2

    obj_spec_00 = Node(min_conn=0)
    obj_spec_01 = Node(min_conn=0)
    assert AggregateAssignmentMatrixGenerator.create(src=[nodes[0], obj_spec_00],
                                                     tgt=[nodes[1], obj_spec_01])._get_max_conn() == 2

    obj_spec_00 = Node(min_conn=2)
    obj_spec_01 = Node(min_conn=0)
    assert AggregateAssignmentMatrixGenerator.create(src=[nodes[0], obj_spec_00],
                                                     tgt=[nodes[1], obj_spec_01])._get_max_conn() == 3


def test_iter_conns(nodes, inf_nodes):
    assert set(AggregateAssignmentMatrixGenerator.create(src=[nodes[0]], tgt=[nodes[1]]).iter_sources()) == {
        (1,),
    }
    assert set(AggregateAssignmentMatrixGenerator.create(src=[inf_nodes[0]], tgt=[nodes[1]]).iter_sources()) == {
        (1,),
    }

    obj = Node([1, 2])
    assert set(AggregateAssignmentMatrixGenerator.create(src=[obj], tgt=[nodes[1]]).iter_sources()) == {
        (1,),
    }

    obj2 = Node([0, 1, 2])
    assert set(AggregateAssignmentMatrixGenerator.create(src=[obj, obj2], tgt=[nodes[1]]).iter_sources()) == {
        (1, 0), (1, 1),
    }

    assert set(AggregateAssignmentMatrixGenerator.create(src=[obj, inf_nodes[0]], tgt=[nodes[1]]).iter_sources()) == {
        (1, 1),
    }

    obj2 = Node([0, 1, 2])
    assert set(AggregateAssignmentMatrixGenerator.create(src=[obj, obj2], tgt=[nodes[1]])._iter_conn_slots([obj, obj2], n=2)) == {
        (1, 1),
    }

    assert set(AggregateAssignmentMatrixGenerator.create(src=[obj, obj2], tgt=[nodes[1]])._iter_conn_slots([obj, obj2])) == {
        (1, 0), (1, 1),
    }
    assert set(AggregateAssignmentMatrixGenerator.create(src=[obj, obj2], tgt=[Node([0, 1, 2])])._iter_conn_slots([obj, obj2])) == {
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1), (2, 2),
    }


def test_iter_order():
    for _ in range(100):
        node1 = Node([1])
        nodes = [Node([0, 1]) for _ in range(5)]
        assert list(AggregateAssignmentMatrixGenerator.create(src=[node1], tgt=nodes)._iter_conn_slots(nodes, is_src=False, n=1)) == \
               [tuple(1 if j == i else 0 for j in range(len(nodes))) for i in reversed(range(len(nodes)))]


def _assert_matrix(gen: AggregateAssignmentMatrixGenerator, matrix_gen, assert_matrix: np.ndarray):
    if isinstance(matrix_gen, np.ndarray):
        matrix_gen = [matrix_gen]
    gen.reset_agg_matrix_cache()
    for _ in range(2):
        agg_matrix = list(gen._agg_matrices(matrix_gen).values())[0]
        assert agg_matrix.shape == assert_matrix.shape
        assert np.all(agg_matrix == assert_matrix)


def test_iter_permuted_conns(nodes):
    gen = AggregateAssignmentMatrixGenerator.create(src=[Node([0, 1]), Node([0, 1])], tgt=[nodes[2]])
    _assert_matrix(gen, gen._iter_matrices([1, 0], [1]), np.array([
        [[1], [0]],
    ]))
    assert gen.count_matrices([1, 0], [1]) == 1

    settings = MatrixGenSettings(src=[nodes[0], nodes[1]], tgt=[nodes[2], nodes[3]])
    gen = AggregateAssignmentMatrixGenerator(settings)
    _assert_matrix(gen, gen._iter_matrices([1, 1], [1, 1]), np.array([
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
    ]))
    assert gen.count_matrices([1, 1], [1, 1]) == 2
    _assert_matrix(gen, gen._iter_matrices((1, 1), (1, 1)), np.array([
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
    ]))
    assert gen.validate_matrix(np.array([[1, 0], [0, 1]]))
    assert not gen.validate_matrix(np.array([[1, 0], [1, 0]]))
    assert not gen.validate_matrix(np.array([[1, 1], [0, 0]]))

    for _ in range(20):
        _assert_matrix(gen, gen._iter_matrices([1, 2], [1, 2]), np.array([
            # [[1, 0], [0, 2]],
            [[0, 1], [1, 1]],
        ]))
        assert gen.count_matrices([1, 2], [1, 2]) == 1

    settings.excluded = [(gen.src[1], gen.tgt[0])]
    gen = AggregateAssignmentMatrixGenerator(settings)
    _assert_matrix(gen, gen._iter_matrices([1, 1], [1, 1]), np.array([
        [[1, 0], [0, 1]],
    ]))
    assert gen.count_matrices([1, 1], [1, 1]) == 1


def test_iter_matrix_zero_conn():
    gen = AggregateAssignmentMatrixGenerator.create(src=[Node([0, 1]), Node([0, 1])], tgt=[Node([0, 1])])
    _assert_matrix(gen, gen._iter_matrices([1, 0], [1]), np.array([
        [[1], [0]],
    ]))
    assert gen.count_matrices([1, 0], [1]) == 1
    _assert_matrix(gen, gen._iter_matrices([0, 0], [0]), np.array([
        [[0], [0]],
    ]))
    assert gen.count_matrices([0, 0], [0]) == 1


def test_iter_edges():
    obj1 = Node(nr_conn_list=[1, 2])
    obj2 = Node(nr_conn_list=[1])
    obj3 = Node(nr_conn_list=[0, 1])
    obj4 = Node(min_conn=1)
    settings = MatrixGenSettings(src=[obj1, obj2], tgt=[obj3, obj4])
    gen = AggregateAssignmentMatrixGenerator(settings)

    for _ in range(10):
        _assert_matrix(gen, gen, np.array([
            [[0, 1], [0, 1]],
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
            [[0, 2], [0, 1]],
            [[1, 1], [0, 1]],
            [[0, 2], [1, 0]],
        ]))
        assert gen.count_all_matrices() == 6
        assert set(gen.iter_conns()) == {
            ((obj1, obj4), (obj2, obj4)),
            ((obj1, obj3), (obj2, obj4)),
            ((obj1, obj4), (obj2, obj3)),
            ((obj1, obj4), (obj1, obj4), (obj2, obj4)),
            ((obj1, obj3), (obj1, obj4), (obj2, obj4)),
            ((obj1, obj4), (obj1, obj4), (obj2, obj3)),
        }

    obj1.rep = False
    gen = AggregateAssignmentMatrixGenerator(settings)
    for _ in range(10):
        assert set(gen.iter_conns()) == {
            ((obj1, obj4), (obj2, obj4)),
            ((obj1, obj3), (obj2, obj4)),
            ((obj1, obj4), (obj2, obj3)),
            ((obj1, obj3), (obj1, obj4), (obj2, obj4)),
        }

    settings.excluded = [(obj2, obj3)]
    gen = AggregateAssignmentMatrixGenerator(settings)
    for _ in range(10):
        assert set(gen.iter_conns()) == {
            ((obj1, obj4), (obj2, obj4)),
            ((obj1, obj3), (obj2, obj4)),
            ((obj1, obj3), (obj1, obj4), (obj2, obj4)),
        }


def test_iter_edges_no_repeat():
    obj1 = Node(nr_conn_list=[1, 2])
    obj2 = Node(nr_conn_list=[1])
    obj3 = Node(nr_conn_list=[0, 1])
    obj4 = Node(min_conn=1, repeated_allowed=False)
    settings = MatrixGenSettings(src=[obj1, obj2], tgt=[obj3, obj4])
    gen = AggregateAssignmentMatrixGenerator(settings)

    for _ in range(10):
        assert set(gen.iter_conns()) == {
            ((obj1, obj4), (obj2, obj4)),
            ((obj1, obj3), (obj2, obj4)),
            ((obj1, obj4), (obj2, obj3)),
            # ((obj1, obj4), (obj1, obj4), (obj2, obj4)),
            ((obj1, obj3), (obj1, obj4), (obj2, obj4)),
            # ((obj1, obj4), (obj1, obj4), (obj2, obj3)),
        }

    obj1.rep = False
    gen = AggregateAssignmentMatrixGenerator(settings)
    for _ in range(10):
        assert set(gen.iter_conns()) == {
            ((obj1, obj4), (obj2, obj4)),
            ((obj1, obj3), (obj2, obj4)),
            ((obj1, obj4), (obj2, obj3)),
            ((obj1, obj3), (obj1, obj4), (obj2, obj4)),
        }

    settings.excluded = [(obj2, obj3)]
    gen = AggregateAssignmentMatrixGenerator(settings)
    for _ in range(10):
        assert set(gen.iter_conns()) == {
            ((obj1, obj4), (obj2, obj4)),
            ((obj1, obj3), (obj2, obj4)),
            ((obj1, obj3), (obj1, obj4), (obj2, obj4)),
        }


def test_node_existence():
    obj1 = Node(nr_conn_list=[1, 2])
    obj2 = Node(nr_conn_list=[1])
    obj3 = Node(nr_conn_list=[0, 1])
    obj4 = Node(min_conn=1)
    exist = NodeExistencePatterns([
        NodeExistence(),
        NodeExistence([True, True], [False, True]),
    ])
    assert hash(exist.patterns[0])
    assert exist.patterns[0]._hash is not None
    assert hash(exist.patterns[1])
    gen = AggregateAssignmentMatrixGenerator.create(src=[obj1, obj2], tgt=[obj3, obj4], existence=exist)

    _assert_matrix(gen, gen, np.array([
        [[0, 1], [0, 1]],
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
        [[0, 2], [0, 1]],
        [[1, 1], [0, 1]],
        [[0, 2], [1, 0]],
    ]))
    gen.reset_agg_matrix_cache()
    assert gen.count_all_matrices() == 6
    assert gen.count_all_matrices(max_by_existence=False) == 8

    matrix = list(gen.get_agg_matrix().values())[0]
    assert gen.validate_matrix(matrix[0, :, :], NodeExistence(tgt_exists=[False, True]))
    assert not gen.validate_matrix(matrix[1, :, :], NodeExistence(tgt_exists=[False, True]))

    assert gen.validate_matrix(matrix[0, :, :], NodeExistence(src_n_conn_override={0: [1]}))
    assert not gen.validate_matrix(matrix[0, :, :], NodeExistence(src_n_conn_override={0: [2]}))
    assert gen.validate_matrix(matrix[3, :, :], NodeExistence(src_n_conn_override={0: [2]}))
    assert not gen.validate_matrix(matrix[0, :, :], NodeExistence(tgt_n_conn_override={1: [1]}))
    assert gen.validate_matrix(matrix[1, :, :], NodeExistence(tgt_n_conn_override={1: [1]}))

    matrix2 = list(gen.get_agg_matrix(cache=True).values())[0]
    assert np.all(matrix == matrix2)
    assert gen._load_agg_matrix_from_cache() is not None
    assert gen.count_all_matrices() == 6
    assert gen.count_all_matrices(max_by_existence=False) == 8

    assert list(gen.iter_n_sources_targets())
    assert list(gen.iter_n_sources_targets(existence=exist.patterns[0]))
    assert list(gen.iter_n_sources_targets(existence=exist.patterns[1]))

    assert NodeExistence() == NodeExistence()
    assert NodeExistence(src_exists=[True, False]) != NodeExistence(tgt_exists=[True, False])
    assert NodeExistence(src_exists=[True, False]) == NodeExistence(src_exists=[True, False])
    assert NodeExistence(src_exists=[True, False]) != NodeExistence(src_exists=[True, False], max_src_conn_override=1)
    assert NodeExistence(src_exists=[True, False], max_src_conn_override=2) != \
           NodeExistence(src_exists=[True, False], max_src_conn_override=1)
    d = {NodeExistence(): 1, NodeExistence(src_exists=[True, False]): 6}
    assert NodeExistence() in d
    assert NodeExistence(src_exists=[True, False]) in d
    assert NodeExistence(tgt_exists=[True, False]) not in d

    assert NodeExistence() != NodeExistence(src_n_conn_override={0: [3]})
    assert NodeExistence(src_n_conn_override={0: [3]}) == NodeExistence(src_n_conn_override={0: [3]})
    assert NodeExistence(src_n_conn_override={0: [1, 3]}) != NodeExistence(src_n_conn_override={0: [3]})
    assert NodeExistence(src_n_conn_override={0: [3]}) != NodeExistence(tgt_n_conn_override={0: [3]})
    assert NodeExistence() != NodeExistence(max_src_conn_override=0)
    assert NodeExistence() != NodeExistence(max_src_conn_override=1)
    assert NodeExistence(max_tgt_conn_override=1) != NodeExistence(max_src_conn_override=1)

    assert NodeExistence().get_transpose() == NodeExistence()
    assert NodeExistence(max_tgt_conn_override=1).get_transpose() == NodeExistence(max_src_conn_override=1)
    assert NodeExistence(max_tgt_conn_override=1) == NodeExistence(max_src_conn_override=1).get_transpose()
    assert NodeExistence(src_n_conn_override={0: [3]}, tgt_n_conn_override={0: [2]}).get_transpose() == \
           NodeExistence(src_n_conn_override={0: [2]}, tgt_n_conn_override={0: [3]})


def test_effective_settings():
    settings = MatrixGenSettings(src=[Node([0, 1]), Node(min_conn=2), Node(min_conn=1, max_conn=5)],
                                 tgt=[Node([0], repeated_allowed=False), Node([1], repeated_allowed=False)],
                                 excluded=[(2, 0)])

    eff_settings, src_map, tgt_map = NodeExistence().get_effective_settings(settings)
    eff_settings0 = eff_settings
    assert len(eff_settings.src) == 3
    assert repr(eff_settings.src[:2]) == repr(settings.src[:2])
    assert eff_settings.src[2].conns is None
    assert eff_settings.src[2].min_conns == 1
    assert len(eff_settings.tgt) == 1
    assert tgt_map == {1: 0}
    assert eff_settings.excluded == []

    assert np.all(settings.expand_effective_matrix(
        np.array([[1], [2], [3]]), src_map, tgt_map) == np.array([[0, 1], [0, 2], [0, 3]]))

    eff_settings, src_map, tgt_map = NodeExistence(tgt_n_conn_override={0: [1]}).get_effective_settings(settings)
    assert len(eff_settings.src) == 3
    assert len(eff_settings.tgt) == 2
    assert eff_settings.excluded == [(2, 0)]

    eff_settings, src_map, tgt_map = \
        NodeExistence(src_n_conn_override={1: [0]}, tgt_n_conn_override={0: [1]}).get_effective_settings(settings)
    assert len(eff_settings.src) == 2
    assert src_map == {0: 0, 2: 1}
    assert len(eff_settings.tgt) == 2
    assert eff_settings.excluded == [(1, 0)]

    assert np.all(settings.expand_effective_matrix(
        np.array([[1, 2], [3, 4]]), src_map, tgt_map) == np.array([[1, 2], [0, 0], [3, 4]]))
    assert np.all(settings.expand_effective_matrix(
        np.array([[[1, 2], [3, 4]], [[5, 6], [4, 3]]]), src_map, tgt_map) ==
                  np.array([[[1, 2], [0, 0], [3, 4]], [[5, 6], [0, 0], [4, 3]]]))

    eff_settings, src_map, tgt_map = NodeExistence(max_src_conn_override=3).get_effective_settings(settings)
    assert len(eff_settings.src) == 3
    assert eff_settings.src[0].conns == [0, 1]
    assert eff_settings.src[1].conns == [2, 3]
    assert eff_settings.src[2].conns is None
    assert eff_settings.src[2].min_conns == 1

    eff_settings_map = settings.get_effective_settings()
    assert eff_settings_map[NodeExistence()][0].get_cache_key() == eff_settings0.get_cache_key()

    transpose_settings = settings.get_transpose_settings()
    assert transpose_settings.src == settings.tgt
    assert transpose_settings.tgt == settings.src
    assert transpose_settings.excluded == [(0, 2)]


def test_max_conn_mat():
    settings = MatrixGenSettings(
        src=[Node(min_conn=0), Node(min_conn=0, repeated_allowed=False), Node([0, 1])],
        tgt=[Node(min_conn=0), Node(min_conn=0, repeated_allowed=False), Node([0, 1])],
        excluded=[(0, 1)],
    )
    assert settings.get_max_conn_parallel() == 2
    assert np.all(settings.get_max_conn_matrix() == np.array([
        [2, 0, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]))

    settings.max_conn_parallel = 3
    assert settings.get_max_conn_parallel() == 3
    assert np.all(settings.get_max_conn_matrix() == np.array([
        [3, 0, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]))

    settings = MatrixGenSettings(
        src=[Node(min_conn=0), Node(min_conn=0, repeated_allowed=False), Node([0, 1])],
        tgt=[Node(min_conn=0), Node(min_conn=0, repeated_allowed=False), Node([0, 1])],
        excluded=[(0, 1)], existence=NodeExistencePatterns(patterns=[
            NodeExistence(),
            NodeExistence(max_src_conn_override=3),
        ]),
    )
    assert settings.get_max_conn_parallel() == 2
    assert settings.get_effective_settings()[settings.existence.patterns[1]][0].get_max_conn_parallel() == 3
    gen = AggregateAssignmentMatrixGenerator(settings)
    assert np.max(gen.max_conn_mat[NodeExistence()]) == 2
    assert np.max(gen.max_conn_mat[settings.existence.patterns[1]]) == 3


def test_matrix_all_inf():
    gen = AggregateAssignmentMatrixGenerator.create(src=[Node(min_conn=0), Node(min_conn=0)],
                                                    tgt=[Node(min_conn=0), Node(min_conn=0)])
    assert np.all(gen.max_conn_mat[NodeExistence()] == np.array([
        [2, 2],
        [2, 2],
    ]))
    assert np.all(gen.get_max_src_appear() == [4, 4])
    assert np.all(gen.get_max_tgt_appear() == [4, 4])

    assert list(gen._agg_matrices(gen).values())[0].shape[0] == 81
    gen.reset_agg_matrix_cache()
    assert gen.count_all_matrices() == 81
    gen.get_agg_matrix(cache=True)
    assert gen.count_all_matrices() == 81

    for matrix, _ in gen.iter_matrices():
        assert gen.validate_matrix(matrix)


def test_matrix_all_inf_no_repeat():
    gen = AggregateAssignmentMatrixGenerator.create(
        src=[Node(min_conn=0, repeated_allowed=False), Node(min_conn=0, repeated_allowed=False)],
        tgt=[Node(min_conn=0, repeated_allowed=False), Node(min_conn=0, repeated_allowed=False)],
    )

    _assert_matrix(gen, gen, np.array([  # 16 diff
        [[0, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[0, 0], [1, 0]],
        [[0, 0], [1, 1]],

        [[0, 1], [0, 0]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, 1]],
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
        [[1, 0], [1, 0]],
        [[0, 1], [1, 1]],
        [[1, 0], [1, 1]],

        [[1, 1], [0, 0]],
        [[1, 1], [0, 1]],
        [[1, 1], [1, 0]],
        [[1, 1], [1, 1]],
    ]))
    assert gen.count_all_matrices() == 16

    for matrix, _ in gen.iter_matrices():
        assert gen.validate_matrix(matrix)


def test_matrix_all_inf_no_repeat_23():
    gen = AggregateAssignmentMatrixGenerator.create(src=[Node(min_conn=0, repeated_allowed=False) for _ in range(2)],
                                                    tgt=[Node(min_conn=0, repeated_allowed=False) for _ in range(3)])
    assert gen.settings.get_max_conn_parallel() == 2
    assert np.all(gen.get_max_src_appear() == [3, 3])
    assert np.all(gen.get_max_tgt_appear() == [2, 2, 2])
    assert np.all(gen.max_conn_mat[NodeExistence()] == np.array([
        [1, 1, 1],
        [1, 1, 1],
    ]))
    assert list(gen._iter_conn_slots(gen.src)) == [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 2), (2, 3),
        (3, 0), (3, 1), (3, 2), (3, 3),
    ]

    assert list(gen._iter_conn_slots(gen.tgt, is_src=False, n=2)) == [
        (0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0),
    ]

    _assert_matrix(gen, gen._iter_matrices([0, 2], [0, 0, 2]), np.empty((0, 2, 3)))
    _assert_matrix(gen, gen._iter_matrices([0, 2], [0, 2, 0]), np.empty((0, 2, 3)))
    _assert_matrix(gen, gen._iter_matrices([0, 2], [2, 0, 0]), np.empty((0, 2, 3)))
    assert gen.count_matrices([0, 2], [2, 0, 0]) == 0
    _assert_matrix(gen, gen._iter_matrices([0, 2], [0, 1, 1]), np.array([[[0, 0, 0], [0, 1, 1]]]))
    _assert_matrix(gen, gen._iter_matrices([0, 2], [1, 1, 0]), np.array([[[0, 0, 0], [1, 1, 0]]]))
    _assert_matrix(gen, gen._iter_matrices([0, 2], [1, 0, 1]), np.array([[[0, 0, 0], [1, 0, 1]]]))
    assert gen.count_matrices([0, 2], [1, 1, 0]) == 1

    assert list(gen._iter_conn_slots(gen.tgt, is_src=False, n=3)) == [
        (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 1, 1), (1, 2, 0), (2, 0, 1), (2, 1, 0),
    ]
    _assert_matrix(gen, gen._iter_matrices([0, 3], [0, 1, 2]), np.empty((0, 2, 3)))
    _assert_matrix(gen, gen._iter_matrices([0, 3], [1, 1, 1]), np.array([[[0, 0, 0], [1, 1, 1]]]))
    assert gen.count_matrices([0, 3], [1, 1, 1]) == 1

    gen0 = AggregateAssignmentMatrixGenerator.create(src=[Node([0]), Node(min_conn=0, repeated_allowed=False)],
                                                     tgt=[Node(min_conn=0, repeated_allowed=False) for _ in range(3)])
    assert gen0.settings.get_max_conn_parallel() == 2
    assert np.all(gen0.get_max_src_appear() == [0, 3])
    assert np.all(gen0.get_max_tgt_appear() == [1, 1, 1])
    assert np.all(gen0.max_conn_mat[NodeExistence()] == np.array([
        [0, 0, 0],
        [1, 1, 1],
    ]))
    assert list(gen0._iter_conn_slots(gen0.src)) == [
        (0, 0), (0, 1), (0, 2), (0, 3),
    ]
    _assert_matrix(gen0, gen0._iter_matrices((0, 2), (1, 0, 1)), np.array([[[0, 0, 0], [1, 0, 1]]]))

    _assert_matrix(gen0, gen0, np.array([
        [[0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0]],
        [[0, 0, 0], [0, 1, 1]],
        [[0, 0, 0], [1, 0, 1]],
        [[0, 0, 0], [1, 1, 0]],
        [[0, 0, 0], [1, 1, 1]],
    ]))
    assert gen0.count_matrices((0, 2), (1, 1, 0)) == 1
    assert gen0.count_all_matrices() == 8


def test_count_n_pool_take():
    def _count_no_dup(n_p, n_t):
        return count_n_pool_take(n_p, n_t, (0,)*n_p)[0]  # For cached
        # return count_n_pool_take(n_p, n_t, np.zeros((n_p,), dtype=int))[0]  # For numba

    def _count_dup(n_p, n_t, n_dup):
        return count_n_pool_take(n_p, n_t, n_dup)[0]  # For cached
        # return count_n_pool_take(n_p, n_t, np.array(n_dup))[0]

    for _ in range(1000):
        assert _count_no_dup(0, 0) == 1
        assert _count_no_dup(0, 1) == 0
        assert _count_no_dup(1, 0) == 1
        assert _count_no_dup(1, 1) == 1

        assert [_count_no_dup(2, n) for n in range(3)] == [1, 2, 1]
        assert [_count_no_dup(3, n) for n in range(4)] == [1, 3, 3, 1]
        assert [_count_no_dup(4, n) for n in range(5)] == [1, 4, 6, 4, 1]
        assert [_count_no_dup(5, n) for n in range(6)] == [1, 5, 10, 10, 5, 1]
        assert [_count_no_dup(6, n) for n in range(7)] == [1, 6, 15, 20, 15, 6, 1]

        assert _count_dup(2, 1, (1, 0)) == 1
        assert _count_dup(2, 2, (1, 0)) == 1
        assert _count_dup(3, 1, (1, 0, 0)) == 2
        assert _count_dup(3, 1, (0, 1, 0)) == 2
        assert _count_dup(3, 1, (2, 0, 0)) == 1
        assert _count_dup(3, 2, (1, 0, 0)) == 2
        assert _count_dup(3, 2, (0, 1, 0)) == 2
        assert _count_dup(3, 2, (2, 0, 0)) == 1
        assert _count_dup(3, 3, (0, 1, 0)) == 1

        assert _count_dup(4, 1, (1, 0, 0, 0)) == 3
        assert _count_dup(4, 1, (0, 1, 0, 0)) == 3
        assert _count_dup(4, 1, (0, 0, 1, 0)) == 3
        assert _count_dup(4, 1, (2, 0, 0, 0)) == 2
        assert _count_dup(4, 1, (0, 2, 0, 0)) == 2
        assert _count_dup(4, 1, (3, 0, 0, 0)) == 1
        assert _count_dup(4, 2, (1, 0, 0, 0)) == 4
        assert _count_dup(4, 2, (0, 1, 0, 0)) == 4
        assert _count_dup(4, 2, (0, 0, 1, 0)) == 4
        assert _count_dup(4, 2, (2, 0, 0, 0)) == 2
        assert _count_dup(4, 2, (0, 2, 0, 0)) == 2
        assert _count_dup(4, 2, (3, 0, 0, 0)) == 1
        assert _count_dup(4, 3, (1, 0, 0, 0)) == 3
        assert _count_dup(4, 3, (0, 1, 0, 0)) == 3
        assert _count_dup(4, 3, (0, 0, 1, 0)) == 3
        assert _count_dup(4, 3, (2, 0, 0, 0)) == 2
        assert _count_dup(4, 3, (0, 2, 0, 0)) == 2
        assert _count_dup(4, 3, (3, 0, 0, 0)) == 1
        assert _count_dup(4, 4, (1, 0, 0, 0)) == 1

        assert _count_dup(6, 3, (1, 0, 0, 2, 0, 0)) == 6


def test_count_n_pool_combs():
    def _assert_count_no_dup(n_p, n_t, combinations):
        n, combs = count_n_pool_take(n_p, n_t, (0,)*n_p)  # For cached
        # n, combs = count_n_pool_take(n_p, n_t, np.zeros((n_p,), dtype=np.int64))
        assert combinations.shape == (n, n_p)
        assert np.all(combs == combinations)

    def _assert_count_dup(n_p, n_t, n_dup, combinations):
        n, combs = count_n_pool_take(n_p, n_t, n_dup)  # For cached
        # n, combs = count_n_pool_take(n_p, n_t, np.array(n_dup))
        assert combinations.shape == (n, n_p)
        assert np.all(combs == combinations)

    _assert_count_no_dup(0, 0, np.zeros((1, 0), dtype=bool))
    _assert_count_no_dup(1, 0, np.array([[0]]))
    _assert_count_no_dup(1, 1, np.array([[1]]))

    _assert_count_no_dup(3, 3, np.array([[1, 1, 1]]))
    _assert_count_no_dup(3, 2, np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]]))
    _assert_count_no_dup(5, 3, np.array([
        [1, 1, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [1, 1, 0, 0, 1],
        [1, 0, 1, 1, 0],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 0, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 1, 1, 1],
    ]))

    _assert_count_dup(2, 1, (1, 0), np.array([[1, 0]]))
    _assert_count_dup(2, 2, (1, 0), np.array([[1, 1]]))
    _assert_count_dup(3, 1, (1, 0, 0), np.array([[1, 0, 0], [0, 0, 1]]))
    _assert_count_dup(4, 1, (0, 1, 0, 0), np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))
    _assert_count_dup(4, 2, (0, 2, 0, 0), np.array([[1, 1, 0, 0], [0, 1, 1, 0]]))

    _assert_count_dup(6, 2, (1, 0, 0, 2, 0, 0), np.array([
        [1, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0],
    ]))
    _assert_count_dup(6, 3, (1, 0, 0, 2, 0, 0), np.array([
        [1, 1, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1],
    ]))
    _assert_count_dup(6, 4, (1, 0, 0, 2, 0, 0), np.array([
        [1, 1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1, 1],
        [0, 1, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
    ]))


def test_count_matrices_from_nodes():
    src = [Node(min_conn=0, repeated_allowed=False) for _ in range(2)]
    tgt = [Node(min_conn=0, repeated_allowed=False) for _ in range(3)]
    gen = AggregateAssignmentMatrixGenerator.create(src, tgt)

    assert gen.count_matrices((0, 3), (0, 1, 2)) == 0

    assert gen.count_matrices((1, 1), (1, 1, 0)) == 2
    assert len(list(gen._iter_matrices((1, 1), (1, 1, 0)))) == 2

    assert gen.count_matrices((2, 1), (1, 1, 1)) == 3
    assert len(list(gen._iter_matrices((2, 1), (1, 1, 1)))) == 3

    for src_nr in gen.iter_sources():  # Iterate over source node nrs
        for tgt_nr in gen.iter_targets(n_source=sum(src_nr)):  # Iterate over target node nrs
            n_mat = gen.count_matrices(src_nr, tgt_nr)
            matrices = np.array(list(gen._iter_matrices(src_nr, tgt_nr)))
            assert matrices.shape[0] == n_mat

    matrix = list(gen.get_agg_matrix().values())[0]
    assert matrix.shape[0] == 64
    gen.reset_agg_matrix_cache()
    assert gen.count_all_matrices() == 64
    gen.get_agg_matrix(cache=True)
    assert gen.count_all_matrices() == 64


def test_conditional_existence_n_conns():
    src = [Node(min_conn=1, repeated_allowed=False) for _ in range(2)]
    tgt = [Node([1], repeated_allowed=False) for _ in range(2)]
    gen = AggregateAssignmentMatrixGenerator(MatrixGenSettings(src, tgt))

    assert list(gen._iter_n_sources_targets()) == [
        ((1, 1), (1, 1), NodeExistence()),
    ]
    gen.reset_agg_matrix_cache()
    for _ in range(2):
        assert list(gen.iter_n_sources_targets()) == [
            ((1, 1), (1, 1), NodeExistence()),
        ]

    pat = NodeExistencePatterns([
        NodeExistence(src_exists=[False, False], tgt_exists=[False, False]),
        NodeExistence(src_exists=[True, False], tgt_exists=[True, False]),
        NodeExistence(),
        NodeExistence(src_n_conn_override={0: [1, 2], 1: [0]}, tgt_n_conn_override={0: [0, 1]}),
    ])
    gen = AggregateAssignmentMatrixGenerator(MatrixGenSettings(src, tgt, existence=pat))

    assert list(gen.iter_n_sources_targets()) == [
        ((0, 0), (0, 0), pat.patterns[0]),
        ((1, 0), (1, 0), pat.patterns[1]),
        ((1, 1), (1, 1), pat.patterns[2]),
        ((1, 0), (0, 1), pat.patterns[3]),
        ((2, 0), (1, 1), pat.patterns[3]),
    ]

    existence_patterns = NodeExistencePatterns.get_all_combinations(
        src_is_conditional=[True, True], tgt_is_conditional=[True, True])
    assert len(existence_patterns.patterns) == 16

    existence_patterns = NodeExistencePatterns.get_increasing(
        src_is_conditional=[True, True], tgt_is_conditional=[True, True])
    assert len(existence_patterns.patterns) == 9


def test_conditional_existence():
    src = [Node(min_conn=1, repeated_allowed=False) for _ in range(2)]
    tgt = [Node([1], repeated_allowed=False) for _ in range(2)]
    existence_patterns = NodeExistencePatterns.get_all_combinations(
        src_is_conditional=[True, True], tgt_is_conditional=[True, True])
    assert len(existence_patterns.patterns) == 16

    gen = AggregateAssignmentMatrixGenerator.create(src, tgt, existence=existence_patterns)
    assert gen.settings.get_max_conn_parallel() == 2
    assert np.all(gen.max_conn_mat[NodeExistence()] == np.array([
        [1, 1],
        [1, 1],
    ]))
    assert np.all(gen.get_max_src_appear() == [2, 2])
    assert np.all(gen.get_max_tgt_appear() == [2, 2])

    src_tgt_conns = list(gen.iter_n_sources_targets())
    assert len(src_tgt_conns) == 8

    matrix_map = gen.get_agg_matrix()
    assert len(matrix_map) == 16
    for existence in existence_patterns.patterns:
        assert existence in matrix_map
    all_arrays = list(matrix_map.values())
    assert np.all(np.row_stack(all_arrays) == np.array([
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
        [[1, 1], [0, 0]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, 0]],
        [[0, 0], [1, 1]],
        [[0, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[0, 0], [0, 0]],
    ]))

    def _assert_mask(src_exists, tgt_exists, check_matrix):
        existence = NodeExistence(src_exists=src_exists, tgt_exists=tgt_exists)
        matrix = gen.get_matrix_for_existence(matrix_map, existence)
        assert matrix.shape[0] == check_matrix.shape[0]
        assert np.all(matrix == check_matrix)
        assert np.all([gen.validate_matrix(matrix[i, :, :], NodeExistence(src_exists=src_exists, tgt_exists=tgt_exists))
                       for i in range(matrix.shape[0])])

    _assert_mask(None, None, np.array([
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
    ]))
    _assert_mask([True, True], [True, True], np.array([
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
    ]))
    _assert_mask([True, True], [True, False], np.empty((0, 2, 2)))
    _assert_mask([True, False], [True, False], np.array([
        [[1, 0], [0, 0]],
    ]))
    _assert_mask([False, False], [False, False], np.array([
        [[0, 0], [0, 0]],
    ]))


def test_combination_problem():
    n_tgt = 20
    gen = AggregateAssignmentMatrixGenerator.create(
        src=[Node([1], repeated_allowed=False)], tgt=[Node([0, 1], repeated_allowed=False) for _ in range(n_tgt)])
    matrix = gen.get_agg_matrix()[NodeExistence()]
    assert matrix.shape == (n_tgt, 1, n_tgt)
    assert np.all(matrix[:, 0, :] == np.eye(n_tgt, dtype=int)[::-1, :])


def test_one_to_one_mapping(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    assert gen_one_per_existence.count_all_matrices() == 1


def test_no_tgt():
    gen = AggregateAssignmentMatrixGenerator.create(src=[Node([1])], tgt=[])
    matrix = gen.get_agg_matrix()[NodeExistence()]
    assert matrix.shape[0] == 0

    gen = AggregateAssignmentMatrixGenerator.create(src=[Node([0, 1])], tgt=[])
    matrix = gen.get_agg_matrix()[NodeExistence()]
    assert matrix.shape[0] == 1
    assert gen.get_conn_idx(matrix[0, :, :]) == []


def test_no_src():
    gen = AggregateAssignmentMatrixGenerator.create(src=[], tgt=[Node([1])])
    matrix = gen.get_agg_matrix()[NodeExistence()]
    assert matrix.shape[0] == 0

    gen = AggregateAssignmentMatrixGenerator.create(src=[], tgt=[Node([0, 1])])
    matrix = gen.get_agg_matrix()[NodeExistence()]
    assert matrix.shape[0] == 1
    assert gen.get_conn_idx(matrix[0, :, :]) == []


def test_max_conn_override():
    existence = [
        NodeExistence(),
        NodeExistence(max_src_conn_override=2, max_tgt_conn_override=2),
        NodeExistence(max_src_conn_override=1),
        NodeExistence(max_tgt_conn_override=1),
        NodeExistence(max_src_conn_override=1, max_tgt_conn_override=1),
    ]
    gen = AggregateAssignmentMatrixGenerator.create(
        src=[Node(min_conn=0) for _ in range(2)], tgt=[Node(min_conn=0) for _ in range(2)],
        existence=NodeExistencePatterns(patterns=existence))

    matrix_map = gen.get_agg_matrix()
    assert matrix_map[existence[0]].shape[0] == 81
    assert matrix_map[existence[1]].shape[0] == 26
    assert matrix_map[existence[2]].shape[0] == 9
    assert matrix_map[existence[2]].shape[0] == 9
    assert not np.all(matrix_map[existence[2]] == matrix_map[existence[3]])
    assert matrix_map[existence[4]].shape[0] == 7
    assert np.all(matrix_map[existence[4]] == np.array([
        [[0, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[0, 0], [1, 0]],
        [[0, 1], [0, 0]],
        [[1, 0], [0, 0]],
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
    ]))


def test_one_conn():
    settings = MatrixGenSettings(src=[Node([1, 2, 3])], tgt=[Node(min_conn=1)])
    assert settings.get_max_conn_parallel() == 3
    assert np.all(settings.get_max_conn_matrix() == np.array([
        [3],
    ]))

    gen = AggregateAssignmentMatrixGenerator(settings)
    assert np.all(list(gen.get_agg_matrix().values())[0] == np.array([
        [[1]],
        [[2]],
        [[3]],
    ]))
