import numpy as np
from typing import *
from adsg_core.optimization.assign_enc.matrix import *
from adsg_core.optimization.assign_enc.encoding import *
from adsg_core.optimization.assign_enc.lazy_encoding import *

__all__ = ['AssignmentManagerBase', 'AssignmentManager', 'LazyAssignmentManager']

T = TypeVar('T')


class AssignmentManagerBase:
    """Base class for managing the encoding of assignment problems."""

    def get_random_design_vector(self) -> DesignVector:
        return [dv.get_random() for dv in self.design_vars]

    @staticmethod
    def _is_violated_matrix(matrix: np.ndarray) -> bool:
        """Whether the matrix represents output of the constraint violator imputer"""
        try:
            return matrix[0, 0] == X_INACTIVE_VALUE
        except IndexError:
            return False

    @staticmethod
    def _correct_is_active(vector: DesignVector) -> Tuple[DesignVector, IsActiveVector]:
        """Corrects the design vector (replace -1 with 0) and return is_active vector"""
        corrected_vector = np.array(vector)
        is_active = corrected_vector != X_INACTIVE_VALUE
        corrected_vector[corrected_vector == X_INACTIVE_VALUE] = 0
        return corrected_vector, is_active

    @property
    def encoder(self) -> Encoder:
        raise NotImplementedError

    @property
    def matrix_gen(self) -> AggregateAssignmentMatrixGenerator:
        raise NotImplementedError

    @property
    def design_vars(self):
        raise NotImplementedError

    def get_imputation_ratio(self) -> float:
        """Ratio of the total design space size to the actual amount of possible connections"""
        raise NotImplementedError

    def correct_vector(self, vector: DesignVector, existence: NodeExistence = None) \
            -> Tuple[DesignVector, IsActiveVector]:
        """Correct the design vector so that it matches an existing connection pattern"""
        raise NotImplementedError

    def get_matrix(self, vector: DesignVector, existence: NodeExistence = None) \
            -> Tuple[DesignVector, IsActiveVector, np.ndarray]:
        """Get connection matrix from a given design vector"""
        raise NotImplementedError

    def get_conn_idx(self, vector: DesignVector, existence: NodeExistence = None) \
            -> Tuple[DesignVector, IsActiveVector, Optional[List[Tuple[int, int]]]]:
        """Get node connections for a given design vector"""
        raise NotImplementedError

    def get_conns(self, vector: DesignVector, existence: NodeExistence = None) \
            -> Tuple[DesignVector, IsActiveVector, Optional[List[Tuple[Node, Node]]]]:
        """Get node connections for a given design vector"""
        raise NotImplementedError

    def get_all_design_vectors(self) -> Dict[NodeExistence, np.ndarray]:
        """Returns all possible valid design vectors, with inactive design variables having a value of -1"""
        raise NotImplementedError


class AssignmentManager(AssignmentManagerBase):
    """
    Uses the matrix generator and encoder to implement the conversion between design vector and connection pattern.
    """

    def __init__(self, settings: MatrixGenSettings, encoder: EagerEncoder, cache=True):
        self._matrix_gen = gen = AggregateAssignmentMatrixGenerator(settings)
        self._encoder = encoder

        encoder.matrix = gen.get_agg_matrix(cache=cache)

    @property
    def matrix_gen(self) -> AggregateAssignmentMatrixGenerator:
        return self._matrix_gen

    @property
    def encoder(self) -> EagerEncoder:
        return self._encoder

    @property
    def matrix(self) -> Dict[NodeExistence, np.ndarray]:
        return self._encoder.matrix

    @property
    def design_vars(self):
        return self._encoder.design_vars

    def get_imputation_ratio(self) -> float:
        return self._encoder.get_imputation_ratio()

    def correct_vector(self, vector: DesignVector, existence: NodeExistence = None) \
            -> Tuple[DesignVector, IsActiveVector]:
        """Correct the design vector so that it matches an existing connection pattern"""
        imputed_vector, _ = self._encoder.get_matrix(vector, existence=existence)
        return self._correct_is_active(imputed_vector)

    def get_matrix(self, vector: DesignVector, existence: NodeExistence = None) \
            -> Tuple[DesignVector, IsActiveVector, np.ndarray]:
        """Get connection matrix from a given design vector"""
        imputed_vector, matrix = self._encoder.get_matrix(vector, existence=existence)
        imputed_vector, is_active = self._correct_is_active(imputed_vector)
        return imputed_vector, is_active, matrix

    def get_conn_idx(self, vector: DesignVector, existence: NodeExistence = None) \
            -> Tuple[DesignVector, IsActiveVector, Optional[List[Tuple[int, int]]]]:
        """Get node connections for a given design vector"""

        # Get matrix and impute design vector
        imputed_vector, is_active, matrix = self.get_matrix(vector, existence=existence)
        if self._is_violated_matrix(matrix):
            return imputed_vector, is_active, None

        # Get connections from matrix
        edges_idx = self._matrix_gen.get_conn_idx(matrix)
        return imputed_vector, is_active, edges_idx

    def get_conns(self, vector: DesignVector, existence: NodeExistence = None) \
            -> Tuple[DesignVector, IsActiveVector, Optional[List[Tuple[Node, Node]]]]:
        """Get node connections for a given design vector"""

        # Get matrix and impute design vector
        imputed_vector, is_active, matrix = self.get_matrix(vector, existence=existence)
        if self._is_violated_matrix(matrix):
            return imputed_vector, is_active, None

        # Get connections from matrix
        edges = self._matrix_gen.get_conns(matrix)
        return imputed_vector, is_active, edges

    def get_all_design_vectors(self) -> Dict[NodeExistence, np.ndarray]:
        return self.encoder.padded_design_vectors


class LazyAssignmentManager(AssignmentManagerBase):

    def __init__(self, settings: MatrixGenSettings, encoder: LazyEncoder):
        self._encoder = encoder
        encoder.set_settings(settings)

    @property
    def encoder(self):
        return self._encoder

    @property
    def matrix_gen(self) -> AggregateAssignmentMatrixGenerator:
        return self._encoder.matrix_gen

    @property
    def design_vars(self):
        return self._encoder.design_vars

    def get_imputation_ratio(self) -> float:
        return self._encoder.get_imputation_ratio()

    def correct_vector(self, vector: DesignVector, existence: NodeExistence = None) \
            -> Tuple[DesignVector, IsActiveVector]:
        """Correct the design vector so that it matches an existing connection pattern"""
        imputed_vector, _ = self._encoder.get_matrix(vector, existence=existence)
        return self._correct_is_active(imputed_vector)

    def get_matrix(self, vector: DesignVector, existence: NodeExistence = None) \
            -> Tuple[DesignVector, IsActiveVector, np.ndarray]:
        """Get connection matrix from a given design vector"""
        imputed_vector, matrix = self._encoder.get_matrix(vector, existence=existence)
        imputed_vector, is_active = self._correct_is_active(imputed_vector)
        return imputed_vector, is_active, matrix

    def get_conn_idx(self, vector: DesignVector, existence: NodeExistence = None) \
            -> Tuple[DesignVector, IsActiveVector, Optional[List[Tuple[int, int]]]]:
        """Get node connections for a given design vector"""
        imputed_vector, matrix = self._encoder.get_matrix(vector, existence=existence)
        imputed_vector, is_active = self._correct_is_active(imputed_vector)
        if self._is_violated_matrix(matrix):
            return imputed_vector, is_active, None

        # Get connections from matrix
        edges_idx = self._encoder.get_conn_idx(matrix)
        return imputed_vector, is_active, edges_idx

    def get_conns(self, vector: DesignVector, existence: NodeExistence = None) \
            -> Tuple[DesignVector, IsActiveVector, Optional[List[Tuple[Node, Node]]]]:
        """Get node connections for a given design vector"""
        imputed_vector, matrix = self._encoder.get_matrix(vector, existence=existence)
        imputed_vector, is_active = self._correct_is_active(imputed_vector)
        if self._is_violated_matrix(matrix):
            return imputed_vector, is_active, None

        # Get connections from matrix
        edges = self._encoder.get_conns(matrix)
        return imputed_vector, is_active, edges

    def get_all_design_vectors(self) -> Dict[NodeExistence, np.ndarray]:
        return self._encoder.get_all_design_vectors()
