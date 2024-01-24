import numpy as np
from adsg_core.optimization.assign_enc.matrix import *
from adsg_core.optimization.assign_enc.enumerating.ordinal import *
from adsg_core.optimization.assign_enc.enumerating.recursive import *
from adsg_core.optimization.assign_enc.lazy.imputation.first import *
from adsg_core.tests.assign_enc.test_lazy_encoding import check_lazy_conditionally_active


def test_encoding():
    enc = EnumOrdinalEncoder(LazyFirstImputer())
    enc.set_settings(MatrixGenSettings(src=[Node([1])], tgt=[Node([0])]))
    assert len(enc.design_vars) == 0

    settings = MatrixGenSettings(src=[Node(min_conn=0, repeated_allowed=False) for _ in range(2)],
                                 tgt=[Node([0, 1], repeated_allowed=False) for _ in range(2)])
    enc = EnumOrdinalEncoder(LazyFirstImputer())
    enc.set_settings(settings)
    matrix = enc.matrix_gen.get_agg_matrix()[NodeExistence()]

    assert len(enc.design_vars) == 1
    assert enc.design_vars[0].n_opts == matrix.shape[0]

    assert enc.get_imputation_ratio() == 1.
    assert enc.get_distance_correlation()

    all_x = enc.get_all_design_vectors()[NodeExistence()]
    assert all_x.shape[0] == matrix.shape[0]

    for i in range(matrix.shape[0]):
        dv, mat = enc.get_matrix([i])
        assert dv == [i]
        assert np.all(dv == all_x[i, :])
        assert np.all(mat == matrix[i, :, :])

    check_lazy_conditionally_active(enc)


def test_one_to_one(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    encoder = EnumOrdinalEncoder(LazyFirstImputer())
    encoder.set_settings(gen_one_per_existence.settings)
    assert len(encoder.design_vars) == 0

    assert encoder.get_n_design_points() == 1
    assert encoder.get_imputation_ratio() == 1.2
    assert encoder.get_information_index() == 1
    assert encoder.get_distance_correlation() == 1
    assert encoder.get_all_design_vectors() is not None

    check_lazy_conditionally_active(encoder)


def test_recursive_encoding():
    for n_divide in range(15):
        for n in range(15):
            settings = MatrixGenSettings(src=[Node([1])], tgt=[Node([0, 1]) for _ in range(n)])

            enc = EnumRecursiveEncoder(LazyFirstImputer(), n_divide=n_divide)
            assert enc.n_divide == max(2, n_divide)
            enc.set_settings(settings)
            assert enc.matrix_gen.count_all_matrices() == n
            check_lazy_conditionally_active(enc)

            all_x = enc.get_all_design_vectors()
            assert NodeExistence() in all_x
            assert all_x[NodeExistence()].shape[0] == n
            for x in all_x[NodeExistence()]:
                x_imp, _ = enc.get_matrix(x)
                assert np.all(x_imp == x)

            if n <= 1:
                assert len(enc.design_vars) == 0
                continue
            n_div = max(2, n_divide)
            if n_div >= n:
                assert len(enc.design_vars) == 1
                assert enc.design_vars[0].n_opts == n

                assert enc.get_imputation_ratio() == 1
                assert enc.get_information_index() == (1 if n == 2 else 0)
            else:
                log_ratio = np.log(n)/np.log(n_div)
                assert len(enc.design_vars) == int(np.ceil(log_ratio))
                assert max([dv.n_opts for dv in enc.design_vars]) == n_div

                dv_last = enc._dv_last[NodeExistence()]
                assert enc.design_vars[0].n_opts == dv_last[0]+1

                assert enc.get_imputation_ratio() >= 1
                assert enc.get_information_index() > 0


def test_one_to_one_recursive(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    for n_divide in range(2, 4):
        encoder = EnumRecursiveEncoder(LazyFirstImputer(), n_divide=n_divide)
        encoder.set_settings(gen_one_per_existence.settings)
        assert len(encoder.design_vars) == 0

        assert encoder.get_n_design_points() == 1
        assert encoder.get_imputation_ratio() == 1.2
        assert encoder.get_information_index() == 1
        assert encoder.get_distance_correlation() == 1
        assert encoder.get_all_design_vectors() is not None

        check_lazy_conditionally_active(encoder)
