import copy
import numba
import warnings
import contextlib
import numpy as np
from typing import *
from adsg_core.optimization.assign_enc.matrix import *
from dataclasses import dataclass
from scipy.spatial import distance
from scipy.stats import pearsonr, ConstantInputWarning

warnings.filterwarnings('ignore', category=ConstantInputWarning)

__all__ = ['DiscreteDV', 'DesignVector', 'PartialDesignVector', 'MatrixSelectMask', 'X_INACTIVE_VALUE',
           'IsActiveVector', 'EagerImputer', 'Encoder', 'EagerEncoder', 'filter_design_vectors', 'flatten_matrix',
           'NodeExistence', 'DetectedHighImpRatio']


@dataclass
class DiscreteDV:
    n_opts: int
    conditionally_active: bool

    def get_random(self):
        return np.random.randint(0, self.n_opts)


DesignVector = Union[List[int], np.ndarray]
IsActiveVector = Union[List[bool], np.ndarray]
PartialDesignVector = List[Optional[int]]
MatrixSelectMask = np.ndarray

X_INACTIVE_VALUE = -1


def filter_design_vectors(design_vectors: np.ndarray, vector: PartialDesignVector) -> MatrixSelectMask:
    int_vector = np.array([-1 if val is None else val for val in vector], dtype=int)
    return _filter_design_vectors(design_vectors, int_vector)


@numba.njit()
def _filter_design_vectors(design_vectors: np.ndarray, vector: np.ndarray) -> MatrixSelectMask:
    """Filter matrices along the first dimension given a design vector. Returns a mask of selected matrices."""

    dv_mask = np.ones((design_vectors.shape[0],), dtype=numba.types.bool_)
    for i, value in enumerate(vector):
        if value != -1:
            # Select design vectors that have the targeted value for this design variable
            dv_mask[dv_mask] = np.bitwise_and(dv_mask[dv_mask], design_vectors[dv_mask, i] == value)
    return dv_mask


class EagerImputer:
    """Base class for imputing design vectors to select existing matrices."""

    def __init__(self):
        self._matrix: Optional[Dict[NodeExistence, np.ndarray]] = None
        self._design_vectors: Optional[Dict[NodeExistence, np.ndarray]] = None
        self._design_vectors_zero: Optional[Dict[NodeExistence, np.ndarray]] = None
        self._design_vars: Optional[List[DiscreteDV]] = None

    def initialize(self, matrix: Dict[NodeExistence, np.ndarray], design_vectors: Dict[NodeExistence, np.ndarray],
                   design_vectors_zero: Dict[NodeExistence, np.ndarray], design_vars: List[DiscreteDV]):
        self._matrix = matrix
        self._design_vectors = design_vectors
        self._design_vectors_zero = design_vectors_zero
        self._design_vars = design_vars

    def _get_design_vectors(self, existence: NodeExistence, zero=True) -> np.ndarray:
        dv_map = self._design_vectors_zero if zero else self._design_vectors
        if existence not in dv_map:
            return np.empty((0, 0), dtype=int)
        return dv_map[existence]

    def _get_matrix(self, existence: NodeExistence) -> np.ndarray:
        return EagerEncoder.get_matrix_for_existence(self._matrix, existence)[0]

    def _filter_design_vectors(self, vector: PartialDesignVector, existence: NodeExistence) -> MatrixSelectMask:
        design_vectors = self._get_design_vectors(existence)
        vector = vector[:design_vectors.shape[1]]
        return filter_design_vectors(design_vectors, vector)

    def _return_imputation(self, i_dv: int, existence: NodeExistence) -> Tuple[DesignVector, np.ndarray]:
        design_vectors = self._get_design_vectors(existence, zero=False)
        matrix = self._get_matrix(existence)
        return design_vectors[i_dv, :], matrix[i_dv, :, :]

    def impute(self, vector: DesignVector, existence: NodeExistence, matrix_mask: MatrixSelectMask) \
            -> Tuple[DesignVector, np.ndarray]:
        """Return a new design vector and associated assignment matrix (n_src x n_tgt)"""
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class DetectedHighImpRatio(RuntimeError):
    """Exception thrown when an encoder detects a high imputation ratio during the encoding process"""

    def __init__(self, encoder: Type['Encoder'], imp_ratio: float, msg: str = None):
        self.encoder = encoder
        self.imp_ratio = imp_ratio
        self.msg = msg
        super().__init__()

    def __str__(self):
        msg_str = f': {self.msg}' if self.msg is not None else ''
        return f'{self.__class__.__name__} when encoding "{self.encoder!s}" ({self.imp_ratio}){msg_str}'

    def __repr__(self):
        return f'{self.__class__.__name__}({self.encoder!r}, imp_ratio={self.imp_ratio}, msg={self.msg})'


class Encoder:

    # Set the imputation ratio limit for early detection (only if the encoder supports it)
    # If detected, a DetectedHighImpRatio exception is thrown
    # Use `with_early_detect_high_imp_ratio` as a context manager
    _early_detect_high_imp_ratio = None

    @classmethod
    @contextlib.contextmanager
    def with_early_detect_high_imp_ratio(cls, high_imp_ratio):
        initial_value = cls._early_detect_high_imp_ratio
        cls._early_detect_high_imp_ratio = high_imp_ratio
        try:
            yield high_imp_ratio
        finally:
            cls._early_detect_high_imp_ratio = initial_value

    @property
    def design_vars(self) -> List[DiscreteDV]:
        raise NotImplementedError

    def get_for_imputer(self, imputer):
        raise NotImplementedError

    def get_random_design_vector(self) -> DesignVector:
        return [dv.get_random() for dv in self.design_vars]

    def get_n_design_points(self) -> int:
        return self.calc_n_declared_design_points(self.design_vars)

    @staticmethod
    def calc_n_declared_design_points(design_vars: List[DiscreteDV]) -> int:
        if len(design_vars) == 0:
            return 1
        return int(np.prod([dv.n_opts for dv in design_vars], dtype=float))

    def get_information_index(self) -> float:
        return self.calc_information_index([dv.n_opts for dv in self.design_vars])

    @staticmethod
    def calc_information_index(n_opts: List[int]) -> float:
        n_dv = len(n_opts)
        if n_dv == 0:
            return 1.
        n_combinations = np.prod(n_opts, dtype=float)
        if n_combinations <= 2:
            return 1.
        n_dv_max = np.log2(n_combinations)
        return (n_dv-1.)/(n_dv_max-1.)

    def get_distance_correlation(self, n=100, minimum=False, n_max=2000, n_samples_min=150, limit=.005) -> Optional[float]:
        """Correlation between distances between design vectors and associated matrices;
        optionally returning the minimum across all existence schemes"""
        dv_dist_all = None
        mat_dist_all = None
        n_check = max(2, int(np.floor(n_samples_min/n)))
        sampled_dvs = {}
        n_iter_max = int(np.ceil(n_max / n))
        metric_values, i_iter = None, 0
        n_has_zero = 0
        for i_iter in range(n_iter_max):
            # Extend current samples
            dv_dist, mat_dist, all_zero = self._get_distance_correlation(n=n, sampled_dvs=sampled_dvs)
            if dv_dist_all is None:
                dv_dist_all = dv_dist
                mat_dist_all = mat_dist
                metric_values = np.nan*np.empty((n_iter_max, len(dv_dist) if minimum else 1))
            else:
                dv_dist_all = [dists+dv_dist[i_mat] for i_mat, dists in enumerate(dv_dist_all)]
                mat_dist_all = [dists+mat_dist[i_mat] for i_mat, dists in enumerate(mat_dist_all)]

            # Calculate new metric value
            if minimum:
                for i_mat, dv_dists in enumerate(dv_dist_all):
                    metric_values[i_iter, i_mat] = self._calc_distance_corr(dv_dists, mat_dist_all[i_mat])
            else:
                metric_values[i_iter, 0] = self._calc_distance_corr(dv_dist_all, mat_dist_all)

            # Track the nr of times we have no new design vectors sampled
            if all_zero:
                n_has_zero += 1
                if n_has_zero >= 2:
                    break

            # Check convergence
            if i_iter+1 >= n_check:
                n_samples = [len(samples) for samples in dv_dist_all]
                if (min(n_samples) if minimum else sum(n_samples)) >= n_samples_min:
                    last_values = metric_values[:i_iter+1, :][-n_check:, :]
                    max_diff = np.max(np.max(last_values, axis=0)-np.min(last_values, axis=0))
                    if max_diff <= limit:
                        break

                # Correlation of 1 is usually quickly found
                if np.max(np.abs(metric_values-1)) < 1e-3:
                    break

        metrics_mean = np.mean(metric_values[:i_iter+1, :][-n_check:, :], axis=0)
        return float(np.min(metrics_mean) if minimum else metrics_mean[0])

    def _get_distance_correlation(self, n: int, sampled_dvs=None) -> Tuple[List[List[float]], List[List[float]], bool]:
        if sampled_dvs is None:
            sampled_dvs = {}
        dv_dist_all = []
        mat_dist_all = []
        all_zero = True
        for matrix, des_vectors in self._iter_sampled_dv_mat(n, sampled_dvs):
            if len(des_vectors) <= 1 or len(matrix) <= 1:
                dv_dist_all.append([])
                mat_dist_all.append([])
                continue

            all_zero = False
            des_vectors[des_vectors == X_INACTIVE_VALUE] = 0
            dv_dist_all.append(list(self._calc_internal_dv_distance(des_vectors)[np.triu_indices(des_vectors.shape[0], k=1)]))
            mat_dist_all.append(list(self._calc_internal_distance(matrix)[np.triu_indices(matrix.shape[0], k=1)]))

        return dv_dist_all, mat_dist_all, all_zero

    def _iter_sampled_dv_mat(self, n: int, sampled_dvs: dict):
        raise NotImplementedError

    @staticmethod
    def _calc_internal_dv_distance(arr: np.ndarray) -> np.ndarray:
        return distance.cdist(arr, arr, 'cityblock')  # Manhattan distance

    @staticmethod
    def _calc_internal_distance(arr: np.ndarray) -> np.ndarray:
        return distance.cdist(arr, arr, 'cityblock')  # Manhattan distance

    @staticmethod
    def _calc_distance_corr(dist1, dist2) -> float:
        if isinstance(dist1, list) and len(dist1) > 0 and isinstance(dist1[0], list):
            dist1 = np.array([val for values in dist1 for val in values])
            dist2 = np.array([val for values in dist2 for val in values])
        if len(dist1) != len(dist2):
            raise RuntimeError(f'Different lengths: {len(dist1)} != {len(dist2)}')
        if len(dist1) < 2:
            return 1.

        corr = pearsonr(dist1, dist2).statistic
        return 1. if np.isnan(corr) else corr

    @staticmethod
    def _pad_dv_map(dv_map: Dict[NodeExistence, np.ndarray], n_dv: int) -> Dict[NodeExistence, np.ndarray]:
        padded_dv_map = {}
        for existence, design_vectors in dv_map.items():
            n_extra = n_dv-design_vectors.shape[1]
            if n_extra > 0:
                if design_vectors.shape[0] == 0:
                    design_vectors = np.zeros((0, n_dv), dtype=int)
                else:
                    design_vectors = np.column_stack([
                        design_vectors, np.ones((design_vectors.shape[0], n_extra), dtype=int)*X_INACTIVE_VALUE])
            padded_dv_map[existence] = design_vectors
        return padded_dv_map

    def get_imputation_ratio(self, per_existence=False) -> float:
        """Ratio of the total design space size to the actual amount of possible connections"""
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class EagerEncoder(Encoder):
    """Base class that encodes assignment matrices to discrete design variables."""

    def __init__(self, imputer: EagerImputer, matrix: np.ndarray = None):
        self._matrix = {}
        self._design_vectors = {}
        self._design_vectors_zeros = {}
        self._design_vector_map = {}
        self._design_vars = []
        self._imputer = imputer

        if matrix is not None:
            self.matrix = matrix

    @property
    def matrix(self) -> Dict[NodeExistence, np.ndarray]:
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: MatrixMapOptional):
        if isinstance(matrix, np.ndarray):
            matrix = {NodeExistence(): matrix}

        # Encode separately for each existence mode
        self._matrix = matrix
        self._design_vectors = des_vectors = {existence: self._encode(mat) for existence, mat in matrix.items()}

        self._design_vectors_zeros = des_vectors_zero = {}
        for existence, des_vec in des_vectors.items():
            des_vec_zeros = des_vec.copy()
            des_vec_zeros[des_vec_zeros == X_INACTIVE_VALUE] = 0
            des_vectors_zero[existence] = des_vec_zeros

        # Inactive variables in the design vectors are encoded as -1 (X_INACTIVE_VALUE), however, for lookup
        # (_design_vector_map) they are stored as 0's, as that is what real design vectors would also have
        self._design_vector_map = {existence: {tuple(dv.tolist()): i for i, dv in enumerate(des_vec)}
                                   for existence, des_vec in des_vectors_zero.items()}

        self._design_vars = self.get_design_variables(des_vectors)
        self._imputer.initialize(matrix, self._design_vectors, self._design_vectors_zeros, self._design_vars)

    @property
    def padded_design_vectors(self) -> Dict[NodeExistence, np.ndarray]:
        return self._pad_dv_map(self._design_vectors, len(self.design_vars))

    def set_imputer(self, imputer: EagerImputer):
        self._imputer = imputer
        if self._matrix is not None:
            self._imputer.initialize(self._matrix, self._design_vectors, self._design_vectors_zeros, self._design_vars)

    def get_for_imputer(self, imputer: EagerImputer):
        encoder: EagerEncoder = copy.deepcopy(self)
        encoder.set_imputer(imputer)
        return encoder

    @property
    def n_mat_max(self) -> int:
        return max([matrix.shape[0] for matrix in self._matrix.values()])

    @property
    def design_vars(self) -> List[DiscreteDV]:
        return self._design_vars

    def _iter_sampled_dv_mat(self, n: int, sampled_dvs: dict):
        for existence, matrix in self._matrix.items():
            matrix = self.flatten_matrix(matrix)
            des_vectors = self._design_vectors_zeros[existence]
            if des_vectors.shape[0] == 0:
                continue

            if existence not in sampled_dvs:
                sampled_dvs[existence] = np.ones((matrix.shape[0],), dtype=bool)
            unsampled_mask = sampled_dvs[existence]

            matrix = matrix[unsampled_mask, :]
            des_vectors = des_vectors[unsampled_mask, :]

            if matrix.shape[0] > n:
                i_random = np.random.choice(matrix.shape[0], size=n, replace=False)
                matrix = matrix[i_random, :]
                des_vectors = des_vectors[i_random, :]
                unsampled_mask[np.where(unsampled_mask)[0][i_random]] = False

            yield matrix, des_vectors

    def get_imputation_ratio(self, per_existence=False) -> float:
        n_design_points = self.get_n_design_points()
        n_total = []
        n_valid = []
        for matrix in self._matrix.values():
            n_total.append(n_design_points)
            n_valid.append(matrix.shape[0])
        if per_existence:
            return min([n_tot/n_valid[i] if n_valid[i] > 0 else np.inf for i, n_tot in enumerate(n_total)])
        return sum(n_total)/sum(n_valid) if sum(n_valid) > 0 else np.inf

    def _correct_vector_size(self, vector: DesignVector) -> Tuple[DesignVector, int, int]:
        n_dv = len(self.design_vars)
        vector, n_extra = self.correct_vector_size(n_dv, vector)
        return vector, n_dv, n_extra

    @staticmethod
    def correct_vector_size(n_dv, vector: DesignVector) -> Tuple[DesignVector, int]:
        n_extra = len(vector)-n_dv
        vector = vector[:n_dv]
        return vector, n_extra

    @staticmethod
    def correct_vector_bounds(vector: DesignVector, design_vars: List[DiscreteDV]) -> Tuple[DesignVector, bool]:
        correct_vector = vector.copy()
        is_corrected = False
        for i, dv in enumerate(design_vars):
            if correct_vector[i] < 0:
                correct_vector[i] = 0
                is_corrected = True

            elif correct_vector[i] >= dv.n_opts:
                correct_vector[i] = dv.n_opts-1
                is_corrected = True

        return correct_vector, is_corrected

    def get_matrix(self, vector: DesignVector, existence: NodeExistence = None,
                   matrix_mask: MatrixSelectMask = None) -> Tuple[DesignVector, np.ndarray]:
        """Select a connection matrix (n_src x n_tgt) and impute the design vector if needed.
        Inactive design variables get a value of -1."""

        vector, n_dv, n_extra = self._correct_vector_size(vector)
        extra_vector = [X_INACTIVE_VALUE]*n_extra
        vector, _ = self.correct_vector_bounds(vector, self.design_vars)

        i_mat, existence = self.get_matrix_index(vector, existence=existence, matrix_mask=matrix_mask)
        matrix, existence = self._get_matrix_for_existence(existence)

        # If this existence mode has no matrices, return the zero vector
        if not self._has_existence(existence):
            vector = [X_INACTIVE_VALUE]*len(vector)
            null_matrix = np.zeros((matrix.shape[1], matrix.shape[2]), dtype=int)
            return vector+extra_vector, null_matrix

        # If no matrix can be found, impute
        if i_mat is None:
            if matrix_mask is None:
                matrix_mask = np.ones((matrix.shape[0],), dtype=bool)

            # If the mask filters out all design vectors, there is no need to try imputing
            elif np.all(~matrix_mask):
                null_matrix = matrix[0, :, :]*0
                return [X_INACTIVE_VALUE]*len(vector)+extra_vector, null_matrix

            vector, matrix = self._imputer.impute(vector, existence, matrix_mask)
            if len(vector) < n_dv:
                vector = list(vector) + [X_INACTIVE_VALUE]*(n_dv-len(vector))
            return list(vector)+extra_vector, matrix

        # Design vector directly maps to possible matrix
        vector = self._correct_vector(vector, existence)
        return list(vector)+extra_vector, matrix[i_mat, :, :]

    def is_valid_vector(self, vector: DesignVector, existence: NodeExistence = None,
                        matrix_mask: MatrixSelectMask = None) -> bool:

        vector, n_dv, n_extra = self._correct_vector_size(vector)
        vector, is_corrected = self.correct_vector_bounds(vector, self.design_vars)
        if is_corrected:
            return False

        i_mat, existence = self.get_matrix_index(vector, existence=existence, matrix_mask=matrix_mask)
        if i_mat is None:
            return False

        vector = list(self._design_vectors[existence][i_mat, :])
        corr_vector = self._correct_vector(vector, existence)
        if not np.all(corr_vector == vector):
            return False
        return True

    def get_matrix_index(self, vector: DesignVector, existence: NodeExistence = None,
                         matrix_mask: MatrixSelectMask = None) -> Tuple[Optional[int], NodeExistence]:
        matrix, existence = self._get_matrix_for_existence(existence)
        if matrix_mask is None:
            matrix_mask = np.ones((matrix.shape[0],), dtype=bool)
        i_mat, = np.where(matrix_mask & self._filter_full_design_vector(matrix, vector, existence=existence))
        if len(i_mat) > 1:
            raise RuntimeError(f'Design vector maps to more than one matrix: {vector}')

        # Only if we find exactly one corresponding matrix, it is indeed a valid design vector
        return i_mat[0] if len(i_mat) == 1 else None, existence

    def _has_existence(self, existence: NodeExistence = None):
        if existence is None:
            existence = NodeExistence()
        return existence in self._matrix

    def _get_matrix_for_existence(self, existence: NodeExistence = None) -> Tuple[np.ndarray, NodeExistence]:
        if len(self._matrix) == 0:
            raise RuntimeError('Matrix not set!')
        return self.get_matrix_for_existence(self._matrix, existence=existence)

    def _correct_vector(self, vector: DesignVector, existence: NodeExistence) -> DesignVector:
        """Set unused design variables to zero (can happen when we are in an existence mode with less design variables
        than the max)"""
        n_dv = 0
        if existence in self._design_vectors:
            n_dv = self._design_vectors[existence].shape[1]

        corrected_vector = vector.copy()
        for i_dv in range(n_dv, len(vector)):
            corrected_vector[i_dv] = X_INACTIVE_VALUE
        return corrected_vector

    @staticmethod
    def get_matrix_for_existence(matrix_map: MatrixMap, existence: NodeExistence = None)\
            -> Tuple[np.ndarray, NodeExistence]:
        if existence is None:
            existence = NodeExistence()
        if existence not in matrix_map:
            first_matrix = list(matrix_map.values())[0]
            return np.empty((0, first_matrix.shape[1], first_matrix.shape[2]), dtype=int), existence
        return matrix_map[existence], existence

    def _filter_full_design_vector(self, matrix: np.ndarray, vector: DesignVector,
                                   existence: NodeExistence = None) -> MatrixSelectMask:
        if existence is None:
            existence = NodeExistence()

        mask = np.zeros((matrix.shape[0],), dtype=bool)
        if existence not in self._design_vector_map:
            return mask
        design_vectors = self._design_vector_map[existence]
        n_dv = self._design_vectors[existence].shape[1]

        i_mat = None
        if n_dv == 0:
            if len(self._matrix[existence]) == 1:
                i_mat = 0
        else:
            i_mat = design_vectors.get(tuple(vector[:n_dv]))

        if i_mat is not None:
            mask[i_mat] = True
        return mask

    @classmethod
    def get_design_variables(cls, design_vectors: Dict[NodeExistence, np.ndarray]) -> List[DiscreteDV]:
        """Convert possible design vectors to design variable definitions"""
        design_vars_list = []
        for des_vectors in design_vectors.values():
            if des_vectors.shape[0] == 0 or des_vectors.shape[1] == 0:
                continue

            # Check if all design vectors are unique
            seen_dvs = set()
            for dv in des_vectors:
                dv_key = tuple(dv.tolist())
                if dv_key in seen_dvs:
                    raise RuntimeError('Not all design vectors are unique!')
                seen_dvs.add(dv_key)

            # Check bounds
            if np.min(des_vectors[des_vectors != X_INACTIVE_VALUE]) != 0:
                raise RuntimeError('Design variables should start at zero!')

            n_opts_max = np.max(des_vectors, axis=0)+1
            is_cond_act = np.any(des_vectors == X_INACTIVE_VALUE, axis=0)
            design_vars_list.append([DiscreteDV(n_opts=n_opts, conditionally_active=is_cond_act[i_dv])
                                     for i_dv, n_opts in enumerate(n_opts_max)])

        # Merge design variables
        design_vars = cls.merge_design_vars(design_vars_list)
        for dv in design_vars:
            if dv.n_opts <= 1:
                raise RuntimeError('All design variables must have at least two options')
        return design_vars

    @staticmethod
    def merge_design_vars(design_vars_list: List[List[DiscreteDV]]) -> List[DiscreteDV]:
        if len(design_vars_list) == 0:
            return []
        n_dv_max = max([len(dvs) for dvs in design_vars_list])
        n_opts = np.zeros((len(design_vars_list), n_dv_max), dtype=int)
        is_cond_act = np.ones(n_opts.shape, dtype=bool)
        for i, dvs in enumerate(design_vars_list):
            n_opts[i, :len(dvs)] = [dv.n_opts for dv in dvs]
            is_cond_act[i, :len(dvs)] = [dv.conditionally_active for dv in dvs]

        n_opts_max = np.max(n_opts, axis=0)
        is_cond_act = np.any(is_cond_act, axis=0)
        return [DiscreteDV(n_opts=n, conditionally_active=is_cond_act[i]) for i, n in enumerate(n_opts_max)]

    @staticmethod
    def flatten_matrix(matrix: np.ndarray) -> np.ndarray:
        return flatten_matrix(matrix)

    @staticmethod
    def normalize_design_vectors(design_vectors: np.ndarray, remove_gaps=True) -> np.ndarray:
        """Move lowest values to 0 and eliminate value gaps."""

        # Check if there are no design vectors defined
        if design_vectors.shape[0] == 0:
            return design_vectors

        # Move to zero
        design_vectors = design_vectors.copy()
        if not remove_gaps:
            for i_dv in range(design_vectors.shape[1]):
                is_act_mask = design_vectors[:, i_dv] != X_INACTIVE_VALUE
                min_value = np.min(design_vectors[is_act_mask, i_dv])
                design_vectors[is_act_mask, i_dv] -= min_value

        # Remove gaps (also moves to zero)
        else:
            for i_dv in range(design_vectors.shape[1]):
                des_var = design_vectors[:, i_dv].copy()
                unique_values = np.sort(np.unique(des_var[des_var != X_INACTIVE_VALUE]))
                for i_unique, value in enumerate(unique_values):
                    design_vectors[des_var == value, i_dv] = i_unique

        # Remove design variables with not enough options
        no_opts_mask = np.max(design_vectors, axis=0) == 0
        design_vectors = design_vectors[:, ~no_opts_mask]

        return design_vectors

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        """
        Encode a matrix of size n_patterns x n_src x n_tgt as discrete design variables.
        Returns the list of design vectors for each matrix in a n_patterns x n_dv array.
        Assumes values range between 0 and the number of options per design variable, or -1 for inactive variables.
        """
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


def flatten_matrix(matrix: np.ndarray) -> np.ndarray:
    """Helper function that flattens matrices in the higher dimensions: n_mat x n_src x n_tgt --> n_mat x n_src*n_tgt"""
    return matrix.reshape(matrix.shape[0], np.prod(matrix.shape[1:]))
