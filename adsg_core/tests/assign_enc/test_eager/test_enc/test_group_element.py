import numpy as np
from adsg_core.optimization.assign_enc.matrix import *
from adsg_core.optimization.assign_enc.eager.imputation.first import *
from adsg_core.optimization.assign_enc.eager.imputation.closest import *
from adsg_core.optimization.assign_enc.eager.encodings.group_element import *
from adsg_core.tests.assign_enc.test_encoder import check_conditionally_active


def test_element_grouper():
    matrix = np.array([
        [[0, 1, 0],
         [1, 1, 0]],
        [[2, 0, 0],
         [1, 1, 2]],
        [[1, 1, 0],
         [1, 1, 2]],
        [[1, 1, 0],
         [1, 2, 2]],
    ])

    encoder = ElementGroupedEncoder(FirstImputer(), matrix)
    encoder.matrix = matrix

    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0, -1],
        [2, -1],
        [1,  0],
        [1,  1],
    ]))

    assert [dv.n_opts for dv in encoder.design_vars] == [3, 2]
    assert encoder.get_n_design_points() == 3*2
    assert encoder.get_imputation_ratio() == 1.5

    encoder.normalize_within_group = False
    encoder.matrix = matrix
    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0, -1],
        [2, -1],
        [1,  0],
        [1,  1],
    ]))

    check_conditionally_active(encoder)


def test_encoder():
    # Combinations with replacement
    gen = AggregateAssignmentMatrixGenerator.create(src=[Node([3])], tgt=[Node(min_conn=0) for _ in range(3)])

    matrix = gen.get_agg_matrix()[NodeExistence()]
    encoder = ElementGroupedEncoder(ClosestImputer(), matrix)
    assert encoder.n_mat_max == 10

    assert len(encoder.design_vars) == 2
    assert encoder.design_vars[0].n_opts == 4
    assert encoder.design_vars[1].n_opts == 4

    dv_seen = set()
    for i0 in range(encoder.design_vars[0].n_opts):
        for i1 in range(encoder.design_vars[1].n_opts):
            dv, mat = encoder.get_matrix([i0, i1])
            assert encoder.is_valid_vector(dv)
            assert gen.validate_matrix(mat)
            if not np.all(dv == np.array([i0, i1])):
                assert not encoder.is_valid_vector([i0, i1])
            dv_seen.add(tuple(dv))

    assert len(dv_seen) == 10

    check_conditionally_active(encoder)


def test_one_to_one(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    encoder = ElementGroupedEncoder(ClosestImputer())
    encoder.matrix = gen_one_per_existence.get_agg_matrix()
    assert len(encoder.design_vars) == 0

    assert encoder.get_n_design_points() == 1
    assert encoder.get_imputation_ratio() == 1.2
    assert encoder.get_information_index() == 1
    assert encoder.get_distance_correlation() == 1

    check_conditionally_active(encoder)


def test_conn_idx_grouper():
    for by_src in [True, False]:
        matrix = np.array([
            [[0, 1, 0],
             [1, 1, 0]],
            [[2, 0, 0],
             [1, 1, 2]],
            [[1, 1, 0],
             [1, 1, 2]],
            [[1, 1, 0],
             [1, 2, 2]],
        ])

        encoder = ConnIdxGroupedEncoder(FirstImputer(), matrix, by_src=by_src)
        encoder.matrix = matrix

        if by_src:
            assert np.all(list(encoder._design_vectors.values())[0] == np.array([
                [1, -1, -1],
                [0,  0, -1],
                [0,  1,  1],
                [0,  1,  0],
            ]))
        else:
            assert np.all(list(encoder._design_vectors.values())[0] == np.array([
                [1, -1, -1],
                [0,  0, -1],
                [0,  1,  0],
                [0,  1,  1],
            ]))

        assert [dv.n_opts for dv in encoder.design_vars] == [2, 2, 2]
        assert encoder.get_n_design_points() == 8
        assert encoder.get_imputation_ratio() == 2

        encoder.matrix = np.array([
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        ])
        if by_src:
            assert np.all(list(encoder._design_vectors.values())[0] == np.array([
                [0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1],
            ]))
        else:
            assert np.all(list(encoder._design_vectors.values())[0] == np.array([
                [0, 0], [0, 1], [1, 0], [2, 0], [1, 1], [2, 1],
            ]))
        assert [dv.n_opts for dv in encoder.design_vars] == [3, 2]
        assert encoder.get_n_design_points() == 6
        assert encoder.get_imputation_ratio() == 1

        check_conditionally_active(encoder)


def test_conn_idx_encoder():
    for binary in [False, True]:
        for by_src in [True, False]:
            # Permutations
            src = [Node([1], repeated_allowed=False) for _ in range(3)]
            tgt = [Node([1], repeated_allowed=False) for _ in range(3)]
            gen = AggregateAssignmentMatrixGenerator.create(src=src, tgt=tgt)

            matrix = gen.get_agg_matrix()[NodeExistence()]
            encoder = ConnIdxGroupedEncoder(ClosestImputer(), matrix, by_src=by_src, binary=binary)
            assert encoder.n_mat_max == 6

            assert len(encoder.design_vars) == (3 if binary else 2)
            if binary:
                continue

            assert encoder.design_vars[0].n_opts == 3
            assert encoder.design_vars[1].n_opts == 2

            dv_seen = set()
            for i0 in range(encoder.design_vars[0].n_opts):
                for i1 in range(encoder.design_vars[1].n_opts):
                    dv, mat = encoder.get_matrix([i0, i1])
                    assert encoder.is_valid_vector(dv)
                    assert dv == [i0, i1]
                    assert gen.validate_matrix(mat)
                    dv_seen.add(tuple(dv))

            assert len(dv_seen) == 6

            check_conditionally_active(encoder)


def test_one_to_one_conn_idx(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    for binary in [False, True]:
        encoder = ConnIdxGroupedEncoder(ClosestImputer(), binary=binary)
        encoder.matrix = gen_one_per_existence.get_agg_matrix()
        assert len(encoder.design_vars) == 0

        assert encoder.get_n_design_points() == 1
        assert encoder.get_imputation_ratio() == 1.2
        assert encoder.get_information_index() == 1
        assert encoder.get_distance_correlation() == 1

        check_conditionally_active(encoder)
