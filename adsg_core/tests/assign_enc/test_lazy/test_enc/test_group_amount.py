import numpy as np
from adsg_core.optimization.assign_enc.matrix import *
from adsg_core.optimization.assign_enc.lazy.imputation.delta import *
from adsg_core.optimization.assign_enc.lazy.encodings.group_amount import *
from adsg_core.optimization.assign_enc.lazy.imputation.constraint_violation import *
from adsg_core.tests.assign_enc.test_lazy_encoding import check_lazy_conditionally_active


def test_flat_amount_encoder():
    dvs = FlatLazyAmountEncoder().encode(2, 2, [((0, 1), (0, 1)), ((1, 1), (0, 2)), ((1, 1), (2, 0)), ((1, 2), (2, 1))],
                                         NodeExistence())
    assert len(dvs) == 1
    assert dvs[0].n_opts == 4


def test_flat_connection_encoder():
    settings = MatrixGenSettings(src=[Node([0, 1, 2]), Node(min_conn=0)], tgt=[Node([0, 1]), Node(min_conn=1)])
    assert settings.get_max_conn_parallel() == 2
    assert np.all(settings.get_max_conn_matrix() == np.array([
        [1, 2],
        [1, 2],
    ]))

    encoder = LazyAmountFirstEncoder(LazyConstraintViolationImputer(), FlatLazyAmountEncoder(), FlatLazyConnectionEncoder())
    encoder.set_settings(settings)

    n_tgt_n_src = list(encoder._n_matrix_map[NodeExistence()].keys())
    assert len(n_tgt_n_src) == 17
    assert ((2, 2), (1, 3)) in n_tgt_n_src

    assert len(encoder.design_vars) == 2
    assert encoder.design_vars[0].n_opts == 17

    matrices_max = encoder._n_matrix_map[NodeExistence()][(2, 2), (1, 3)]
    assert matrices_max.shape[0] == 2
    assert encoder.design_vars[1].n_opts == 2

    dv, matrix = encoder.get_matrix([15, 0])
    assert np.all(dv == [15, 0])
    assert np.all(matrix == matrices_max[0, :, :])
    assert np.all(matrix == np.array([[1, 1], [0, 2]]))
    dv, matrix = encoder.get_matrix([15, 1])
    assert np.all(dv == [15, 1])
    assert np.all(matrix == matrices_max[1, :, :])

    assert encoder.get_n_design_points() == 34
    assert encoder.get_imputation_ratio() == 34/21
    assert encoder.get_distance_correlation()

    check_lazy_conditionally_active(encoder)


def test_flat_connection_encoder_multi_max():
    encoder = LazyAmountFirstEncoder(LazyConstraintViolationImputer(), FlatLazyAmountEncoder(), FlatLazyConnectionEncoder())
    encoder.set_settings(MatrixGenSettings(
        src=[Node(min_conn=0, repeated_allowed=False) for _ in range(2)],
        tgt=[Node([1], repeated_allowed=False) for _ in range(3)],
    ))

    assert len(list(encoder._n_matrix_map[NodeExistence()].keys())) == 4
    assert encoder.matrix_gen.count_all_matrices() == 8

    assert len(encoder.design_vars) == 2
    assert encoder.design_vars[0].n_opts == 4
    assert encoder.design_vars[1].n_opts == 3

    check_lazy_conditionally_active(encoder)


def test_total_amount_encoder():
    dvs = TotalLazyAmountEncoder().encode(2, 2, [((0, 1), (0, 1)), ((1, 1), (0, 2)), ((1, 1), (2, 0)), ((1, 2), (2, 1))],
                                          NodeExistence())
    assert len(dvs) == 2
    assert dvs[0].n_opts == 3
    assert dvs[1].n_opts == 2

    dvs = TotalLazyAmountEncoder().encode(2, 2, [((0, 1), (0, 1)), ((1, 1), (0, 2))], NodeExistence())
    assert len(dvs) == 1
    assert dvs[0].n_opts == 2

    encoder = LazyAmountFirstEncoder(LazyConstraintViolationImputer(), TotalLazyAmountEncoder(), FlatLazyConnectionEncoder())
    encoder.set_settings(MatrixGenSettings(src=[Node([0, 1, 2]), Node(min_conn=0)], tgt=[Node([0, 1]), Node(min_conn=1)]))

    n_tgt_n_src = list(encoder._n_matrix_map[NodeExistence()].keys())
    assert len(n_tgt_n_src) == 17
    assert len(encoder.design_vars) == 3
    assert encoder.design_vars[0].n_opts == 5
    assert encoder.design_vars[1].n_opts == 6

    dv, matrix = encoder.get_matrix([0, 5, 0])
    assert np.all(dv == [0, 5, 0])
    assert np.all(matrix == -1)

    dv, matrix = encoder.get_matrix([3, 2, 0])
    assert np.all(matrix == np.array([[1, 1], [0, 2]]))

    check_lazy_conditionally_active(encoder)


def test_source_amount_encoder():
    dvs = SourceLazyAmountEncoder().encode(2, 2, [((0, 1), (0, 1)), ((1, 1), (0, 2)), ((1, 1), (2, 0)), ((1, 2), (2, 1))],
                                           NodeExistence())
    assert len(dvs) == 3
    assert dvs[0].n_opts == 2
    assert dvs[1].n_opts == 2
    assert dvs[2].n_opts == 2

    encoder = LazyAmountFirstEncoder(LazyDeltaImputer(), SourceLazyAmountEncoder(), FlatLazyConnectionEncoder())
    encoder.set_settings(MatrixGenSettings(
        src=[Node([0, 1, 2]), Node(min_conn=0)],
        tgt=[Node([0, 1]), Node(min_conn=1)],
        existence=NodeExistencePatterns([
            NodeExistence(),
            NodeExistence(tgt_exists=[True, False]),
        ]),
    ))

    n_tgt_n_src = list(encoder._n_matrix_map[NodeExistence()].keys())
    assert len(n_tgt_n_src) == 17
    assert len(encoder.design_vars) == 4
    assert encoder.design_vars[0].n_opts == 3
    assert encoder.design_vars[1].n_opts == 4
    assert encoder.design_vars[2].n_opts == 2

    dv, matrix = encoder.get_matrix([0, 0, 0, 0])
    assert np.all(dv == [0, 0, -1, 0])
    assert np.all(matrix == np.array([[0, 0], [0, 1]]))

    dv, matrix = encoder.get_matrix([1, 0, 0, 0], existence=encoder.existence_patterns.patterns[1])
    assert np.all(dv == [1, -1, -1, -1])
    assert np.all(matrix == np.array([[1, 0], [0, 0]]))
    dv, matrix = encoder.get_matrix([1, 0, 0, 1], existence=encoder.existence_patterns.patterns[1])
    assert np.all(dv == [1, -1, -1, -1])
    assert np.all(matrix == np.array([[1, 0], [0, 0]]))

    check_lazy_conditionally_active(encoder)


def test_source_target_amount_encoder():
    dvs = SourceTargetLazyAmountEncoder().encode(
        2, 2, [((0, 1), (0, 1)), ((1, 1), (0, 2)), ((1, 1), (2, 0)), ((1, 2), (2, 1))], NodeExistence())
    assert len(dvs) == 3
    assert dvs[0].n_opts == 2
    assert dvs[1].n_opts == 2
    assert dvs[2].n_opts == 2

    encoder = LazyAmountFirstEncoder(LazyConstraintViolationImputer(), SourceTargetLazyAmountEncoder(), FlatLazyConnectionEncoder())
    encoder.set_settings(MatrixGenSettings(src=[Node([0, 1, 2]), Node(min_conn=0)], tgt=[Node([0, 1]), Node(min_conn=1)]))

    n_tgt_n_src = list(encoder._n_matrix_map[NodeExistence()].keys())
    assert len(n_tgt_n_src) == 17
    assert len(encoder.design_vars) == 4
    assert encoder.design_vars[0].n_opts == 3
    assert encoder.design_vars[1].n_opts == 4
    assert encoder.design_vars[2].n_opts == 2

    check_lazy_conditionally_active(encoder)


def test_filter_dvs():
    encoder = LazyAmountFirstEncoder(LazyConstraintViolationImputer(), FlatLazyAmountEncoder(), FlatLazyConnectionEncoder())
    encoder.set_settings(MatrixGenSettings(src=[Node([1], repeated_allowed=False), Node([1], repeated_allowed=False)],
                                           tgt=[Node([1], repeated_allowed=False), Node([1], repeated_allowed=False)]))

    n_tgt_n_src = list(encoder._n_matrix_map[NodeExistence()].keys())
    assert len(n_tgt_n_src) == 1

    assert len(encoder.design_vars) == 1
    assert encoder.design_vars[0].n_opts == 2

    dv, matrix = encoder.get_matrix([0])
    assert np.all(dv == [0])
    assert np.all(matrix == np.array([[1, 0], [0, 1]]))

    dv, matrix = encoder.get_matrix([1])
    assert np.all(dv == [1])
    assert np.all(matrix == np.array([[0, 1], [1, 0]]))

    check_lazy_conditionally_active(encoder)


def test_covering_partitioning():
    encoder = LazyAmountFirstEncoder(LazyDeltaImputer(), FlatLazyAmountEncoder(), FlatLazyConnectionEncoder())
    settings = MatrixGenSettings(src=[Node(min_conn=0, repeated_allowed=False) for _ in range(2)],
                                 tgt=[Node(min_conn=1, repeated_allowed=False) for _ in range(4)])
    assert np.all(settings.get_max_conn_matrix() == np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]))
    encoder.set_settings(settings)

    n_tgt_n_src = list(encoder._n_matrix_map[NodeExistence()].keys())
    assert len(n_tgt_n_src) == 48
    assert encoder.matrix_gen.count_all_matrices() == 81
    assert encoder.get_n_design_points() >= 81

    check_lazy_conditionally_active(encoder)


def test_one_to_one(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    g = gen_one_per_existence
    encoder = LazyAmountFirstEncoder(LazyDeltaImputer(), FlatLazyAmountEncoder(), FlatLazyConnectionEncoder())
    encoder.set_settings(g.settings)
    assert len(encoder.design_vars) == 0

    assert encoder.get_n_design_points() == 1
    assert encoder.get_imputation_ratio() == 1.2
    assert encoder.get_information_index() == 1
    assert encoder.get_distance_correlation() == 1

    for i, existence in enumerate(gen_one_per_existence.existence_patterns.patterns):
        dv, mat = encoder.get_matrix([], existence=existence)
        assert dv == []
        assert mat.shape[0] == (len(gen_one_per_existence.src) if i not in [3, 7] else 0)

    check_lazy_conditionally_active(encoder)
