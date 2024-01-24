import copy
import itertools
import numpy as np
from typing import *
from adsg_core.optimization.assign_enc.matrix import *
from adsg_core.optimization.assign_enc.encoding import *

__all__ = ['LazyImputer', 'LazyEncoder', 'DesignVector', 'DiscreteDV', 'NodeExistence', 'X_INACTIVE_VALUE',
           'QuasiLazyEncoder', 'DetectedHighImpRatio']


class LazyImputer:
    """Base class for imputing design vectors to select existing matrices, works with lazy encoding."""

    def __init__(self):
        self._matrix_gen: Optional[AggregateAssignmentMatrixGenerator] = None
        self._existence_des_vars: Optional[Dict[NodeExistence, List[DiscreteDV]]] = None
        self._des_vars: Optional[List[DiscreteDV]] = None
        self._decode_func = None
        self._impute_cache = {}

    def initialize(self, matrix_gen: AggregateAssignmentMatrixGenerator,
                   existence_des_vars: Dict[NodeExistence, List[DiscreteDV]], design_vars: List[DiscreteDV],
                   decode_func):
        self._matrix_gen = matrix_gen
        self._existence_des_vars = existence_des_vars
        self._des_vars = design_vars
        self._decode_func = decode_func
        self._impute_cache = {}

    def _decode(self, vector: DesignVector, existence: NodeExistence) -> Optional[Tuple[DesignVector, np.ndarray]]:
        return self._decode_func(vector, existence)

    def _get_des_vars(self, existence: NodeExistence) -> List[DiscreteDV]:
        return self._existence_des_vars.get(existence, self._des_vars)

    def impute(self, vector: DesignVector, matrix: Optional[np.ndarray], existence: NodeExistence,
               tried_vectors: set = None) -> Tuple[DesignVector, np.ndarray]:
        """Returns the imputed design vector and associated connection matrix"""

        # Check in cache
        cache_key = hash((tuple(vector), hash(existence)))
        if cache_key in self._impute_cache:
            return self._impute_cache[cache_key]

        def _validate(mat):
            if mat is None:
                return False
            return self._matrix_gen.validate_matrix(mat, existence=existence)

        # If none of the nodes exist, return empty matrix
        n_src, n_tgt = len(self._matrix_gen.src), len(self._matrix_gen.tgt)
        if existence.none_exists(n_src, n_tgt):
            imputed_vector = [X_INACTIVE_VALUE]*len(vector)
            imputed_matrix = np.zeros((n_src, n_tgt), dtype=int)

        else:
            # Try to modify vector such that it adheres to the src and tgt exist masks
            if matrix is not None:
                matrix = matrix.copy()
                matrix[~existence.src_exists_mask(n_src), :] = 0
                matrix[:, ~existence.tgt_exists_mask(n_tgt)] = 0
            if matrix is not None and _validate(matrix):
                imputed_vector, imputed_matrix = vector, matrix

            else:  # Custom imputation
                if tried_vectors is None:
                    tried_vectors = set()
                tried_vectors.add(tuple(vector))

                vector = np.array(vector)
                imputed_vector, imputed_matrix = self._impute(vector, matrix, existence, _validate, tried_vectors)

        # Update cache
        self._impute_cache[cache_key] = (imputed_vector, imputed_matrix)
        return imputed_vector, imputed_matrix

    def _impute(self, vector: DesignVector, matrix: Optional[np.ndarray], existence: NodeExistence,
                validate: Callable[[np.ndarray], bool], tried_vectors: Set[Tuple[int, ...]]) \
            -> Tuple[DesignVector, np.ndarray]:
        """Returns the imputed design vector and associated connection matrix"""
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class LazyEncoder(Encoder):
    """Encoder that skips the matrix-generation step (so it might be better suited for large numbers of connections) by
    relying on two-way design variable encoders."""

    _n_mc_imputation_ratio = 10000

    def __init__(self, imputer: LazyImputer):
        self._matrix_gen: Optional[AggregateAssignmentMatrixGenerator] = None
        self._existence_design_vars: Optional[Dict[NodeExistence, List[DiscreteDV]]] = None
        self._design_vars: Optional[List[DiscreteDV]] = None
        self._imputer = imputer
        self._empty_matrix = None

    def set_settings(self, settings: MatrixGenSettings):
        self._matrix_gen = gen = AggregateAssignmentMatrixGenerator(settings)

        self._encode_prepare()
        self._existence_design_vars = existence_dvs = \
            {existence: self._encode(existence) for existence in gen.iter_existence()}
        self._design_vars = dvs = self._merge_design_vars(list(existence_dvs.values()))
        for i, dv in enumerate(dvs):
            if dv.n_opts < 2:
                raise RuntimeError(f'All design variables must have at least 2 options: {i} has {dv.n_opts} opts')

        self._imputer.initialize(self._matrix_gen, self._existence_design_vars, self._design_vars, self._decode)
        self._empty_matrix = None

    def set_imputer(self, imputer: LazyImputer):
        self._imputer = imputer
        if self._matrix_gen is not None:
            self._imputer.initialize(self._matrix_gen, self._existence_design_vars, self._design_vars, self._decode)

    def get_for_imputer(self, imputer: LazyImputer):
        encoder: LazyEncoder = copy.deepcopy(self)
        encoder.set_imputer(imputer)
        return encoder

    @property
    def src(self):
        return self._matrix_gen.src

    @property
    def tgt(self):
        return self._matrix_gen.tgt

    @property
    def n_src(self):
        return len(self.src)

    @property
    def n_tgt(self):
        return len(self.tgt)

    @property
    def ex(self):
        return self._matrix_gen.ex

    @property
    def existence_patterns(self):
        return self._matrix_gen.existence_patterns

    @property
    def matrix_gen(self) -> AggregateAssignmentMatrixGenerator:
        return self._matrix_gen

    @property
    def design_vars(self) -> List[DiscreteDV]:
        return self._design_vars

    def get_empty_matrix(self) -> np.ndarray:
        if self._empty_matrix is None:
            self._empty_matrix = np.zeros((self.n_src, self.n_tgt), dtype=int)
        return self._empty_matrix.copy()

    def _iter_sampled_dv_mat(self, n: int, sampled_dvs: dict):
        n_sample = n*5
        for existence in self._matrix_gen.iter_existence():
            if existence not in sampled_dvs:
                sampled_dvs[existence] = set()
            sampled = sampled_dvs[existence]

            # Try if there is an encoder-specific way to generate random design vectors and matrices
            random_dv_mat = self._generate_random_dv_mat(n, existence)
            n_sampled = 0
            matrix, des_vectors = [], []
            if random_dv_mat is not None:
                for i_try in range(2):
                    if i_try > 0:
                        random_dv_mat = self._generate_random_dv_mat(n-n_sampled, existence)
                    dv_random, mat_random = random_dv_mat
                    for i, dv in enumerate(dv_random):
                        dv_hash = hash(tuple(dv))
                        if dv_hash in sampled:
                            continue
                        sampled.add(dv_hash)

                        # valid = self._matrix_gen.validate_matrix(mat_random[i], existence=existence)
                        # if not valid:
                        #     raise RuntimeError(f'Generated a non-valid matrix (random_dv_mat): {self!r}')

                        matrix.append(mat_random[i].ravel())
                        des_vectors.append(dv)
                        n_sampled += 1
                        if n_sampled >= n:
                            break

                    if n_sampled >= n or dv_random.shape[0] < n:
                        break

            else:
                # Otherwise, generate random design vectors and decode
                des_vars, n_dv, _, extra_dv = self._get_des_vars_n_extra(existence)
                random_dvs = np.empty((n_sample, n_dv), dtype=int)
                for i_dv, dv in enumerate(des_vars):
                    random_dvs[:, i_dv] = dv.n_opts*np.random.random((n_sample,))

                for i in range(n*5):
                    dv_ = random_dvs[i, :]
                    dv_init_hash = hash(tuple(dv_))
                    if dv_init_hash in sampled:
                        continue

                    # Get matrix without imputation to save time
                    dv, mat, _ = self._decode_vector(dv_, existence=existence)
                    valid = mat is not None and self._matrix_gen.validate_matrix(mat, existence=existence)

                    dv = list(dv)+extra_dv
                    if not valid:
                        sampled.add(dv_init_hash)
                        continue

                    dv_hash = hash(tuple(dv))
                    if dv_hash in sampled:
                        continue
                    sampled.add(dv_init_hash)
                    sampled.add(dv_hash)

                    matrix.append(mat.ravel())
                    des_vectors.append(dv)
                    n_sampled += 1
                    if n_sampled >= n:
                        break

            matrix = np.array(matrix, dtype=int)
            des_vectors = np.array(des_vectors, dtype=int)
            yield matrix, des_vectors

    def _generate_random_dv_mat(self, n: int, existence: NodeExistence) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Custom implementation for generating valid random design vectors (n x nx) and associated
        matrices (n x n_src x n_tgt). Used for speeding up distance correlation calculations."""

    def get_all_design_vectors(self) -> Dict[NodeExistence, np.ndarray]:
        dv_map = self._get_all_design_vectors(self.existence_patterns.patterns)
        if dv_map is not None:
            return self._pad_dv_map(dv_map, len(self.design_vars))

        # By brute force
        dv_map = {}
        for existence in self.existence_patterns.patterns:
            des_vars = self._existence_design_vars.get(existence, self.design_vars)

            seen_dvs = set()
            design_vectors = []
            for dv in itertools.product(*[list(range(dv.n_opts)) for dv in des_vars]):
                if dv in seen_dvs:
                    continue

                dv, matrix = self.get_matrix(list(dv), existence=existence)
                if tuple(dv) in seen_dvs:
                    continue
                try:
                    if matrix[0, 0] == X_INACTIVE_VALUE:
                        continue
                except IndexError:
                    continue
                design_vectors.append(dv)
                seen_dvs.add(tuple(dv))

            if len(design_vectors) == 0:
                dv_map[existence] = np.zeros((0, len(des_vars)), dtype=int)
            else:
                dv_map[existence] = np.array(design_vectors)

        return self._pad_dv_map(dv_map, len(self.design_vars))

    def _get_all_design_vectors(self, patterns: List[NodeExistence]) -> Optional[Dict[NodeExistence, np.ndarray]]:
        """Implement if it is possible to generate all possible design vectors"""

    def get_imputation_ratio(self, per_existence=False, use_real_matrix=True) -> float:
        if use_real_matrix:
            n_design_points = self.get_n_design_points()
            n_total = []
            n_valid = []
            for matrix in self._matrix_gen.get_agg_matrix(cache=True).values():
                n_total.append(n_design_points)
                n_valid.append(matrix.shape[0])
            if per_existence:
                return min([n_tot/n_valid[i] if n_valid[i] > 0 else np.inf for i, n_tot in enumerate(n_total)])
            return sum(n_total)/sum(n_valid) if sum(n_valid) > 0 else np.inf

        n_sample = self._n_mc_imputation_ratio
        n_total = []
        n_valid = []
        for existence in self._matrix_gen.iter_existence():
            n_total_ex = 0
            n_valid_ex = 0
            if n_sample > self.get_n_design_points():
                # Exhaustive sampling
                for dv in itertools.product(*[list(range(dv.n_opts)) for dv in self.design_vars]):
                    n_total_ex += 1
                    if self.is_valid_vector(list(dv), existence=existence):
                        n_valid_ex += 1

            else:  # Monte Carlo sampling
                for _ in range(n_sample):
                    n_total_ex += 1
                    if self.is_valid_vector(self.get_random_design_vector(), existence=existence):
                        n_valid_ex += 1
            n_total.append(n_total_ex)
            n_valid.append(n_valid_ex)

        if per_existence:
            return min([n_tot/n_valid[i] if n_valid[i] > 0 else np.inf for i, n_tot in enumerate(n_total)])
        return sum(n_total)/sum(n_valid) if sum(n_valid) > 0 else np.inf

    def _correct_vector_size(self, vector: DesignVector, existence: NodeExistence = None) \
            -> Tuple[DesignVector, int, int, List[DiscreteDV], NodeExistence]:
        if existence is None:
            existence = NodeExistence()

        des_vars = self._existence_design_vars.get(existence, self.design_vars)
        n_dv = len(des_vars)
        vector, n_extra = EagerEncoder.correct_vector_size(n_dv, vector)
        return vector, n_dv, n_extra, des_vars, existence

    def _get_des_vars_n_extra(self, existence: NodeExistence):
        des_vars = self._existence_design_vars.get(existence, self.design_vars)
        n_dv = len(des_vars)
        n_extra = len(self._design_vars)-n_dv
        extra_dv = [X_INACTIVE_VALUE]*n_extra
        return des_vars, n_dv, n_extra, extra_dv

    def get_matrix(self, vector: DesignVector, existence: NodeExistence = None) -> Tuple[DesignVector, np.ndarray]:
        """Select a connection matrix (n_src x n_tgt) and impute the design vector if needed.
        Inactive design variables get a value of -1."""
        vector, extra_vector, existence, matrix, valid = self._get_validate_matrix(vector, existence=existence)
        if valid:
            return list(vector)+extra_vector, matrix

        vector, matrix = self._impute(vector, matrix, existence)
        return list(vector)+extra_vector, matrix

    def _impute(self, vector, matrix, existence: NodeExistence) -> Tuple[DesignVector, np.ndarray]:
        return self._imputer.impute(vector, matrix, existence)

    def _get_validate_matrix(self, vector: DesignVector, existence: NodeExistence = None) \
            -> Tuple[List[int], List[int], NodeExistence, np.ndarray, bool]:
        vector, n_dv, n_extra, dvs, existence = self._correct_vector_size(vector, existence)
        extra_vector = [X_INACTIVE_VALUE]*n_extra
        vector, _ = EagerEncoder.correct_vector_bounds(vector, dvs)

        # Decode matrix
        vector, matrix, existence = self._decode_vector(vector, existence)

        # Validate matrix
        valid = matrix is not None and self._matrix_gen.validate_matrix(matrix, existence=existence)
        return vector, extra_vector, existence, matrix, valid

    def is_valid_vector(self, vector: DesignVector, existence: NodeExistence = None):

        original_vector = vector
        vector, n_dv, n_extra, dvs, existence = self._correct_vector_size(vector, existence)
        vector, is_corrected = EagerEncoder.correct_vector_bounds(vector, dvs)
        if is_corrected:
            return False

        vector, matrix, existence = self._decode_vector(vector, existence)
        if matrix is None:
            return False

        if not self._matrix_gen.validate_matrix(matrix, existence=existence):
            return False

        if n_extra > 0 and sum(original_vector[n_dv:]) != 0:
            return False
        return True

    def _decode_vector(self, vector: DesignVector, existence: NodeExistence = None):
        if existence is None:
            existence = NodeExistence()
        results = self._decode(vector, existence=existence)
        if results is None:
            return vector, None, existence

        vector, matrix = results
        return vector, matrix, existence

    def get_conn_idx(self, matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Convert matrix to edge tuples"""
        edges = []
        for i_src in range(matrix.shape[0]):
            for j_tgt in range(matrix.shape[1]):
                for _ in range(matrix[i_src, j_tgt]):
                    edges.append((i_src, j_tgt))
        return edges

    def get_conns(self, matrix: np.ndarray) -> List[Tuple[Node, Node]]:
        """Convert matrix to edge tuples"""
        edges = []
        for i_src in range(matrix.shape[0]):
            src = self.src[i_src]
            for j_tgt in range(matrix.shape[1]):
                tgt = self.tgt[j_tgt]
                for _ in range(matrix[i_src, j_tgt]):
                    edges.append((src, tgt))
        return edges

    @staticmethod
    def _filter_dvs(dvs: List[DiscreteDV]) -> Tuple[List[DiscreteDV], np.ndarray, int]:
        """Return design variables that have at least 2 options"""
        valid_dv_mask = np.zeros((len(dvs),), dtype=bool)
        valid_dv = []
        for i, dv in enumerate(dvs):
            if dv.n_opts >= 2:
                valid_dv_mask[i] = True
                valid_dv.append(dv)

        return valid_dv, valid_dv_mask, len(dvs)

    @staticmethod
    def _unfilter_dvs(vector: DesignVector, i_valid_dv: np.ndarray, n: int) -> DesignVector:
        all_vector = np.zeros((n,), dtype=int)
        all_vector[i_valid_dv] = vector[:n]
        return all_vector

    @staticmethod
    def _merge_design_vars(design_vars_list: List[List[DiscreteDV]]) -> List[DiscreteDV]:
        return EagerEncoder.merge_design_vars(design_vars_list)

    def _encode_prepare(self):
        pass

    def _encode(self, existence: NodeExistence) -> List[DiscreteDV]:
        """Encode the assignment problem (given by src and tgt nodes) directly to design variables"""
        raise NotImplementedError

    def _decode(self, vector: DesignVector, existence: NodeExistence) -> Optional[Tuple[DesignVector, np.ndarray]]:
        """Return the connection matrix as would be encoded by the given design vector"""
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class QuasiLazyEncoder(LazyEncoder):
    """Lazy encoder that does depend on generating all matrices, however does not explicitly define all design vectors"""

    def __init__(self, imputer: LazyImputer):
        super().__init__(imputer)
        self._matrix_map = None

    def _encode_prepare(self):
        self._matrix_map = self._matrix_gen.get_agg_matrix(cache=True)

    def _get_matrix(self, existence: NodeExistence) -> Optional[np.ndarray]:
        return self._matrix_map.get(existence)

    def _encode(self, existence: NodeExistence) -> List[DiscreteDV]:
        matrix = self._get_matrix(existence)
        if matrix is None:
            raise RuntimeError(f'Encoding an unknown existence scheme: {existence!r}')
        return self._encode_matrix(matrix, existence)

    def _decode(self, vector: DesignVector, existence: NodeExistence) -> Optional[Tuple[DesignVector, np.ndarray]]:
        matrix = self._get_matrix(existence)
        if matrix is None or matrix.shape[0] == 0:
            null_matrix = np.zeros((self.n_src, self.n_tgt), dtype=int)
            return vector, null_matrix

        return self._decode_matrix(vector, matrix, existence)

    def _generate_random_dv_mat(self, n: int, existence: NodeExistence) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        design_vars = self._existence_design_vars.get(existence, self.design_vars)
        matrix = self._get_matrix(existence)
        if matrix is None or matrix.shape[0] == 0:
            null_dvs = np.zeros((0, len(design_vars)), dtype=int)
            null_matrices = np.zeros((0, self.n_src, self.n_tgt), dtype=int)
            return null_dvs, null_matrices

        return self._do_generate_random_dv_mat(n, existence, matrix, design_vars)

    def _get_all_design_vectors(self, patterns: List[NodeExistence]) -> Optional[Dict[NodeExistence, np.ndarray]]:
        dv_map = {}
        for existence in patterns:
            design_vars = self._existence_design_vars.get(existence, self.design_vars)
            matrix = self._get_matrix(existence)
            if matrix is None or matrix.shape[0] == 0:
                null_dvs = np.zeros((0, len(design_vars)), dtype=int)
                dv_map[existence] = null_dvs
                continue

            design_vectors = self._do_get_all_design_vectors(existence, matrix, design_vars)
            if design_vectors is None:
                return
            dv_map[existence] = design_vectors
        return dv_map

    def _encode_matrix(self, matrix: np.ndarray, existence: NodeExistence) -> List[DiscreteDV]:
        raise NotImplementedError

    def _decode_matrix(self, vector: DesignVector, matrix: np.ndarray, existence: NodeExistence) \
            -> Optional[Tuple[DesignVector, np.ndarray]]:
        raise NotImplementedError

    def _do_generate_random_dv_mat(self, n: int, existence: NodeExistence, matrix: np.ndarray,
                                   design_vars: List[DiscreteDV]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        pass

    def _do_get_all_design_vectors(self, existence: NodeExistence, matrix: np.ndarray, design_vars: List[DiscreteDV]) \
            -> Optional[np.ndarray]:
        pass

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError
