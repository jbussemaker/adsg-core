import pytest
import itertools
import numpy as np
from adsg_core.optimization.assign_enc.encoding import *
from adsg_core.optimization.assign_enc.eager.imputation.first import *
from adsg_core.optimization.assign_enc.eager.encodings.grouped_base import *
from adsg_core.tests.assign_enc.test_encoder import check_conditionally_active


class ProductEncoder(GroupedEncoder):

    def _get_grouping_values(self, matrix: np.ndarray) -> np.ndarray:
        n_var = int(np.ceil(np.log2(matrix.shape[0])))
        design_vectors = np.array(list(itertools.product(*[[0, 1] for _ in range(n_var)]))[:matrix.shape[0]])
        for i_var in reversed(range(1, n_var)):
            design_vectors[:, i_var:] += np.array([2*design_vectors[:, i_var-1]]).T
        return design_vectors


def test_half_grouped_encoder():
    matrix = np.random.randint(0, 3, (11, 4, 3))
    encoder = ProductEncoder(FirstImputer(), matrix)
    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0, 0, 0, 0, 0, 0, 0, 0,  1,  1,  1],
        [0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1],
        [0, 0, 1, 1, 0, 0, 1, 1,  0,  0,  1],
        [0, 1, 0, 1, 0, 1, 0, 1,  0,  1, -1],
    ]).T)
    assert len(encoder.design_vars) == 4
    assert all([dv.n_opts == 2 for dv in encoder.design_vars])
    assert encoder.get_imputation_ratio() == (2**4)/11

    check_conditionally_active(encoder)


def test_early_detect_high_imp_ratio():
    matrix = np.random.randint(0, 3, (2**10+1, 4, 3))
    encoder = ProductEncoder(FirstImputer(), matrix)
    assert len(encoder.design_vars) == 11
    assert encoder.get_imputation_ratio() == (2**11)/(2**10+1)

    with Encoder.with_early_detect_high_imp_ratio(1.9):
        assert Encoder._early_detect_high_imp_ratio == 1.9
        with pytest.raises(DetectedHighImpRatio):
            ProductEncoder(FirstImputer(), matrix)
    assert Encoder._early_detect_high_imp_ratio is None

    values = np.array([np.arange(4**4+1)]).T
    values_base4 = ProductEncoder.convert_to_base(values, base=4)
    assert np.prod(np.max(values_base4, axis=0)+1)/values.shape[0] == (4**4*2)/(4**4+1)
    with Encoder.with_early_detect_high_imp_ratio(1.9):
        with pytest.raises(DetectedHighImpRatio):
            ProductEncoder.convert_to_base(values, base=4)

    with Encoder.with_early_detect_high_imp_ratio(100):
        with pytest.raises(DetectedHighImpRatio):
            ProductEncoder.convert_to_base(values, base=4, n_declared_start=1e6)

    try:
        assert Encoder._early_detect_high_imp_ratio is None
        with Encoder.with_early_detect_high_imp_ratio(100):
            assert Encoder._early_detect_high_imp_ratio == 100
            raise RuntimeError
    except RuntimeError:
        pass
    assert Encoder._early_detect_high_imp_ratio is None


def test_no_group_normalize():
    matrix = np.random.randint(0, 3, (11, 4, 3))
    encoder = ProductEncoder(FirstImputer(), matrix, normalize_within_group=False)
    dvs = list(encoder._design_vectors.values())[0]
    assert np.all(dvs == np.array([
        [0, 0, 0, 0, 0, 0, 0, 0,  1,  1,  1],
        [0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1],
        [0, 0, 1, 1, 2, 2, 3, 3,  2,  2,  3],
        [0, 1, 2, 3, 2, 3, 4, 5,  2,  3, -1],
    ]).T)
    assert len(encoder.design_vars) == 4
    assert [dv.n_opts for dv in encoder.design_vars] == [2, 2, 4, 6]


def test_convert_base():
    matrix = np.random.randint(0, 3, (11, 4, 3))
    encoder = ProductEncoder(FirstImputer(), matrix, normalize_within_group=False)
    dvs = list(encoder._design_vectors.values())[0]
    dvs_base2 = GroupedEncoder.convert_to_base(dvs, base=2)
    dvs_base2_ref = np.array([
        # 0 1  2  2  3  3  3
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 1],
        [0, 1, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 0, 1],
        [1, -1, 1, 0, 0, 1, 0],
        [1, -1, 1, 0, 0, 1, 1],
        [1, -1, 1, 1, -1, -1, -1],
    ])
    assert np.all(dvs_base2 == dvs_base2_ref)

    assert np.all(GroupedEncoder.convert_to_base(dvs[::-1, :], base=2) == dvs_base2_ref[::-1, :])


class EvenNonUniqueGrouper(GroupedEncoder):

    def _get_grouping_values(self, matrix: np.ndarray) -> np.ndarray:
        n_var = int(np.ceil(np.log2(matrix.shape[0])))
        design_vectors = np.array(list(itertools.product(*[[0, 1] for _ in range(n_var)]))[:matrix.shape[0]])
        for i_var in reversed(range(1, n_var)):
            design_vectors[:, i_var:] += np.array([2*design_vectors[:, i_var-1]]).T

        null_design_vectors = np.zeros((design_vectors.shape[0], design_vectors.shape[1]*2))
        null_design_vectors[:, ::2] = design_vectors
        return null_design_vectors


def test_skip_des_var():
    matrix = np.random.randint(0, 3, (11, 4, 3))
    encoder = EvenNonUniqueGrouper(FirstImputer(), matrix=matrix)
    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0, 0, 0, 0, 0, 0, 0, 0,  1,  1,  1],
        [0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1],
        [0, 0, 1, 1, 0, 0, 1, 1,  0,  0,  1],
        [0, 1, 0, 1, 0, 1, 0, 1,  0,  1, -1],
    ]).T)


def test_index_grouped_encoder():
    matrix = np.random.randint(0, 3, (11, 4, 3))
    encoder = GroupByIndexEncoder(FirstImputer(), n_groups=2, matrix=matrix)
    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0, 0, 0, 0, 0, 0, 0, 0,  1,  1,  1],
        [0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1],
        [0, 0, 1, 1, 0, 0, 1, 1,  0,  0,  1],
        [0, 1, 0, 1, 0, 1, 0, 1,  0,  1, -1],
    ]).T)

    encoder = GroupByIndexEncoder(FirstImputer(), n_groups=3, matrix=matrix)
    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0,  1,  1],
        [0, 0, 0, 1, 1, 1, 2, 2, 2, -1, -1],
        [0, 1, 2, 0, 1, 2, 0, 1, 2,  0,  1],
    ]).T)

    encoder = GroupByIndexEncoder(FirstImputer(), n_groups=4, matrix=matrix)
    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2],
        [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2],
    ]).T)


class PreprocessedIndexGroupedEncoder(GroupedEncoder):

    def _get_grouping_values(self, matrix: np.ndarray) -> np.ndarray:
        n_mat = matrix.shape[0]
        des_var_values = np.zeros((n_mat, 2))

        n_half = n_mat // 2
        des_var_values[n_half:, 0] = 1
        des_var_values[:n_half, 1] = np.arange(0, n_half)
        des_var_values[n_half:, 1] = np.arange(0, n_mat-n_half)

        return des_var_values


def test_grouped_encoder_by_sub_matrix_idx():
    matrix = np.random.randint(0, 3, (11, 4, 3))
    encoder = PreprocessedIndexGroupedEncoder(FirstImputer(), matrix)
    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5],
    ]).T)


def test_ordinal_base_grouping():
    assert np.all(GroupedEncoder.group_by_values(np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [2, 0],
    ]), ordinal_base=2) == np.array([
        [0, 0,  0],
        [0, 0,  1],
        [0, 1, -1],
        [1, 0, -1],
    ]))
