import numpy as np
from adsg_core.optimization.assign_enc.matrix import *
from adsg_core.optimization.assign_enc.encoding import *
from adsg_core.optimization.assign_enc.eager.imputation.first import *
from adsg_core.optimization.assign_enc.eager.encodings import *
from adsg_core.tests.assign_enc.test_encoder import check_conditionally_active


def test_encoder():
    for _ in range(10):
        matrix = np.random.randint(0, 5, (10, 4, 3))
        n_tot = np.sum(flatten_matrix(matrix), axis=1)

        encoder = AmountFirstGroupedEncoder(FirstImputer(), TotalAmountGrouper(), OneVarLocationGrouper())
        encoder.matrix = matrix

        assert len(encoder.design_vars) <= 2
        assert encoder.get_imputation_ratio() >= 1.

        assert encoder.get_distance_correlation()
        assert encoder.get_distance_correlation(minimum=True)

        n_tot_unique = np.unique(n_tot)
        assert encoder.design_vars[0].n_opts == len(n_tot_unique)
        n_tot_max = max([np.sum(n_tot == n_val) for n_val in n_tot_unique])
        if len(encoder.design_vars) == 2:
            assert encoder.design_vars[1].n_opts == n_tot_max

        check_conditionally_active(encoder)


def test_source_amount_grouper():
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

    encoder = AmountFirstGroupedEncoder(FirstImputer(), SourceAmountGrouper(), OneVarLocationGrouper())
    encoder.matrix = matrix

    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0, -1, -1],
        [1,  0,  0],
        [1,  0,  1],
        [1,  1, -1],
    ]))

    encoder.normalize_within_group = False
    encoder.matrix = matrix
    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0, -1, -1],
        [1,  0,  0],
        [1,  0,  1],
        [1,  1, -1],
    ]))

    check_conditionally_active(encoder)


def test_source_amount_flat_grouper():
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

    encoder = AmountFirstGroupedEncoder(FirstImputer(), SourceAmountFlattenedGrouper(), OneVarLocationGrouper())
    encoder.matrix = matrix

    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0, -1],
        [1,  0],
        [1,  1],
        [2, -1],
    ]))

    encoder.normalize_within_group = False
    encoder.matrix = matrix
    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0, -1],
        [1,  0],
        [1,  1],
        [2, -1],
    ]))

    check_conditionally_active(encoder)


def test_target_amount_grouper():
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

    encoder = AmountFirstGroupedEncoder(FirstImputer(), TargetAmountGrouper(), OneVarLocationGrouper())
    encoder.matrix = matrix

    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0, -1],
        [2, -1],
        [1,  0],
        [1,  1],
    ]))

    encoder.normalize_within_group = False
    encoder.matrix = matrix
    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0, -1],
        [2, -1],
        [1,  0],
        [1,  1],
    ]))

    check_conditionally_active(encoder)


def test_target_amount_flat_grouper():
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

    encoder = AmountFirstGroupedEncoder(FirstImputer(), TargetAmountFlattenedGrouper(), OneVarLocationGrouper())
    encoder.matrix = matrix

    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0],
        [3],
        [1],
        [2],
    ]))

    encoder.normalize_within_group = False
    encoder.matrix = matrix
    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0],
        [3],
        [1],
        [2],
    ]))

    check_conditionally_active(encoder)


def test_flat_index_loc_grouper():
    matrix = np.array([
        [[0, 1, 0],
         [1, 1, 0]],
        [[0, 0, 1],
         [1, 1, 0]],
        [[0, 0, 1],
         [1, 0, 1]],
        [[0, 0, 1],
         [0, 1, 1]],
        [[0, 0, 1],
         [0, 0, 2]],
    ])
    assert len(np.unique(np.sum(flatten_matrix(matrix), axis=1))) == 1

    loc_grouper = FlatIndexLocationGrouper()
    encoder = AmountFirstGroupedEncoder(FirstImputer(), TotalAmountGrouper(), loc_grouper)
    encoder.matrix = matrix

    assert len(encoder.design_vars) == 3
    assert np.all(loc_grouper.get_grouping_values(matrix) == np.array([
        [1, 3, 4],
        [2, 3, 4],
        [2, 3, 5],
        [2, 4, 5],
        [2, 5, 5],
    ]))
    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0, -1, -1],
        [1,  0,  0],
        [1,  0,  1],
        [1,  1, -1],
        [1,  2, -1],
    ]))

    encoder.normalize_within_group = False
    encoder.matrix = matrix
    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0, -1, -1],
        [1,  0,  0],
        [1,  0,  1],
        [1,  1, -1],
        [1,  2, -1],
    ]))

    check_conditionally_active(encoder)


def test_rel_flat_index_loc_grouper():
    matrix = np.array([
        [[0, 1, 0],
         [1, 1, 0]],
        [[0, 0, 1],
         [1, 1, 0]],
        [[0, 0, 1],
         [1, 0, 1]],
        [[0, 0, 1],
         [0, 1, 1]],
        [[0, 0, 1],
         [0, 0, 2]],
    ])
    assert len(np.unique(np.sum(flatten_matrix(matrix), axis=1))) == 1

    loc_grouper = RelFlatIndexLocationGrouper()
    encoder = AmountFirstGroupedEncoder(FirstImputer(), TotalAmountGrouper(), loc_grouper)
    encoder.matrix = matrix

    assert len(encoder.design_vars) == 3
    assert np.all(loc_grouper.get_grouping_values(matrix) == np.array([
        [1, 2, 1],
        [2, 1, 1],
        [2, 1, 2],
        [2, 2, 1],
        [2, 3, 0],
    ]))
    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0, -1, -1],
        [1,  0,  0],
        [1,  0,  1],
        [1,  1, -1],
        [1,  2, -1],
    ]))

    encoder.normalize_within_group = False
    encoder.matrix = matrix
    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0, -1, -1],
        [1,  0,  0],
        [1,  0,  1],
        [1,  1, -1],
        [1,  2, -1],
    ]))

    check_conditionally_active(encoder)


def test_coord_index_loc_grouper():
    matrix = np.array([
        [[0, 1, 0],
         [1, 1, 0]],
        [[0, 0, 1],
         [1, 1, 0]],
        [[0, 0, 1],
         [1, 0, 1]],
        [[0, 0, 1],
         [0, 1, 1]],
        [[0, 0, 1],
         [0, 0, 2]],
    ])
    assert len(np.unique(np.sum(flatten_matrix(matrix), axis=1))) == 1

    loc_grouper = CoordIndexLocationGrouper()
    encoder = AmountFirstGroupedEncoder(FirstImputer(), TotalAmountGrouper(), loc_grouper)
    encoder.matrix = matrix

    assert len(encoder.design_vars) == 3
    assert np.all(loc_grouper.get_grouping_values(matrix) == np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 1],
        [0, 1, 0, 2, 0, 1],
    ]))
    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0, -1, -1],
        [1,  0,  0],
        [1,  0,  1],
        [1,  1, -1],
        [1,  2, -1],
    ]))

    encoder.normalize_within_group = False
    encoder.matrix = matrix
    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0, -1, -1],
        [1,  0,  0],
        [1,  0,  1],
        [1,  1, -1],
        [1,  2, -1],
    ]))

    check_conditionally_active(encoder)


def test_rel_coord_index_loc_grouper():
    matrix = np.array([
        [[0, 1, 0],
         [1, 1, 0]],
        [[0, 0, 1],
         [1, 1, 0]],
        [[0, 0, 1],
         [1, 0, 1]],
        [[0, 0, 1],
         [0, 1, 1]],
        [[0, 0, 1],
         [0, 0, 2]],
    ])
    assert len(np.unique(np.sum(flatten_matrix(matrix), axis=1))) == 1

    loc_grouper = RelCoordIndexLocationGrouper()
    encoder = AmountFirstGroupedEncoder(FirstImputer(), TotalAmountGrouper(), loc_grouper)
    encoder.matrix = matrix

    assert len(encoder.design_vars) == 3
    assert np.all(loc_grouper.get_grouping_values(matrix) == np.array([
        [0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 2],
        [0, 1, 0, 1, 0, 1],
        [0, 1, 0, 2, 0, 0],
    ]))
    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0, -1, -1],
        [1,  0,  0],
        [1,  0,  1],
        [1,  1, -1],
        [1,  2, -1],
    ]))

    encoder.normalize_within_group = False
    encoder.matrix = matrix
    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0, -1, -1],
        [1,  0,  0],
        [1,  0,  1],
        [1,  1, -1],
        [1,  2, -1],
    ]))

    check_conditionally_active(encoder)


def test_one_to_one(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    encoder = AmountFirstGroupedEncoder(FirstImputer(), TotalAmountGrouper(), OneVarLocationGrouper())
    encoder.matrix = gen_one_per_existence.get_agg_matrix()
    assert len(encoder.design_vars) == 0

    assert encoder.get_n_design_points() == 1
    assert encoder.get_imputation_ratio() == 1.2
    assert encoder.get_information_index() == 1
    assert encoder.get_distance_correlation() == 1

    check_conditionally_active(encoder)
