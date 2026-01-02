import timeit
from adsg_core.optimization.assign_enc.matrix import *
from adsg_core.optimization.assign_enc.selector import *
from adsg_core.optimization.assign_enc.assignment_manager import *


def test_selector():
    src = [Node([1], repeated_allowed=False) for _ in range(2)]
    tgt = [Node([0, 1], repeated_allowed=False) for _ in range(4)]
    selector = EncoderSelector(MatrixGenSettings(src, tgt))
    assert selector._numba_initialized

    assert selector._get_n_mat() == (12, 1)

    assignment_manager = selector._get_best_assignment_manager()
    assert isinstance(assignment_manager, AssignmentManagerBase)

    encoder = assignment_manager.encoder
    assert encoder.get_imputation_ratio() < 10


def test_selector_inf_idx_filter():
    src = [Node([1], repeated_allowed=False) for _ in range(2)]
    tgt = [Node([0, 1], repeated_allowed=False) for _ in range(4)]
    selector = EncoderSelector(MatrixGenSettings(src, tgt))
    selector.n_mat_max_eager = 1
    selector.min_distance_correlation = 1.2
    assert selector._get_n_mat() == (12, 1)

    assignment_manager = selector._get_best_assignment_manager()
    assert isinstance(assignment_manager, AssignmentManagerBase)

    encoder = assignment_manager.encoder
    assert encoder.get_imputation_ratio() < 10


def test_selector_encoding_timeout():
    src = [Node([1], repeated_allowed=False) for _ in range(6)]
    tgt = [Node([0, 1], repeated_allowed=False) for _ in range(6)]
    selector = EncoderSelector(MatrixGenSettings(src, tgt))
    selector.encoding_timeout = .05

    assignment_manager = selector._get_best_assignment_manager()
    assert isinstance(assignment_manager, AssignmentManagerBase)

    encoder = assignment_manager.encoder
    assert encoder.get_imputation_ratio() == 1


def test_selector_cache():
    src = [Node([1], repeated_allowed=False) for _ in range(4)]
    tgt = [Node([0, 1], repeated_allowed=False) for _ in range(4)]
    selector = EncoderSelector(MatrixGenSettings(src, tgt))

    selector.reset_cache()

    s = timeit.default_timer()
    selector.get_best_assignment_manager()
    t1 = timeit.default_timer()-s

    s = timeit.default_timer()
    selector.get_best_assignment_manager()
    t2 = timeit.default_timer()-s
    assert t2 < t1

    s = timeit.default_timer()
    selector.get_best_assignment_manager(cache=False)
    t3 = timeit.default_timer()-s
    assert t3 > t2


def test_one_to_one(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    g = gen_one_per_existence
    selector = EncoderSelector(g.settings)

    assignment_manager = selector._get_best_assignment_manager()
    assert isinstance(assignment_manager, AssignmentManagerBase)

    assert len(assignment_manager.design_vars) == 0


def test_one_to_one_no_patterns(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    EncoderSelector._exclude_pattern_encoders = True
    g = gen_one_per_existence
    selector = EncoderSelector(g.settings)

    assignment_manager = selector._get_best_assignment_manager()
    assert isinstance(assignment_manager, AssignmentManagerBase)

    assert len(assignment_manager.design_vars) == 0
    EncoderSelector._exclude_pattern_encoders = False


def test_selector_no_exist():
    src = [Node([1], repeated_allowed=False)]
    tgt = [Node([2], repeated_allowed=False)]
    selector = EncoderSelector(MatrixGenSettings(src, tgt))
    assert isinstance(selector._get_best_assignment_manager(), AssignmentManagerBase)


def test_selector_single_option():
    src = [Node([1], repeated_allowed=False)]
    tgt = [Node([1], repeated_allowed=False)]
    selector = EncoderSelector(MatrixGenSettings(src, tgt))
    assert isinstance(selector._get_best_assignment_manager(), AssignmentManagerBase)


def test_selector_single_option2():
    src = [Node([1], repeated_allowed=False)]
    tgt = [Node(min_conn=1, repeated_allowed=False)]
    selector = EncoderSelector(MatrixGenSettings(src, tgt))
    assert isinstance(selector._get_best_assignment_manager(), AssignmentManagerBase)


def test_one_opt_one_req():
    src = [Node([1]), Node([0, 1])]
    tgt = [Node([0, 1, 2])]
    selector = EncoderSelector(MatrixGenSettings(src, tgt))

    manager = selector._get_best_assignment_manager()
    all_x = manager.get_all_design_vectors()[NodeExistence()]
    assert all_x.shape == (2, 1)
    matrix_seen = set()
    for x in all_x:
        _, _, matrix = manager.get_matrix(x)
        matrix_seen.add(tuple(matrix.ravel()))
    assert len(matrix_seen) == all_x.shape[0]


def test_all_one_req():
    src = [Node([1]), Node([1])]
    tgt = [Node([1, 2])]
    selector = EncoderSelector(MatrixGenSettings(src, tgt, existence=NodeExistencePatterns([
        NodeExistence(),
        NodeExistence(src_exists=[True, False]),
        NodeExistence(src_exists=[False, True]),
        NodeExistence(src_exists=[False, False]),
    ])))

    manager = selector._get_best_assignment_manager()
    all_x_ext = manager.get_all_design_vectors()
    for existence, all_x in all_x_ext.items():
        matrix_seen = set()
        for x in all_x:
            _, _, matrix = manager.get_matrix(x, existence=existence)
            matrix_seen.add(tuple(matrix.ravel()))
        assert len(matrix_seen) == all_x.shape[0]
