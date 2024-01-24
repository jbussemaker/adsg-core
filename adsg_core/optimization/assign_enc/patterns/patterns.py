import itertools
import numpy as np
from typing import *
from adsg_core.optimization.assign_enc.matrix import *
from adsg_core.optimization.assign_enc.lazy_encoding import *
from adsg_core.optimization.assign_enc.patterns.encoder import *

__all__ = ['CombiningPatternEncoder', 'AssigningPatternEncoder', 'PartitioningPatternEncoder',
           'ConnectingPatternEncoder', 'PermutingPatternEncoder',
           'UnorderedCombiningPatternEncoder']


class CombiningPatternEncoder(PatternEncoderBase):
    """
    Encodes the combining pattern: choosing one of N options.

    Source nodes: 1 node with 1 connection
    Target nodes: N nodes with 0 or 1 connection

    OR ("collapsed mode")
    Source nodes: 1 node with any connection settings
    Target nodes: 1 node with any connection settings

    Encoded as:   1 design variable with n_tgt options (or n_possibilities options)
    """

    def __init__(self, imputer):
        self.is_collapsed = False
        self._min_max_map = {}
        super().__init__(imputer)

    def _matches_pattern(self, effective_settings: MatrixGenSettings, initialize: bool) -> bool:
        src, tgt = effective_settings.src, effective_settings.tgt

        # There may be no excluded edges
        if effective_settings.get_excluded_indices():
            return False

        # Check 1 source node
        if len(src) != 1:
            return False

        # Check if the source node has 1 connection
        if src[0].conns != [1]:
            # Check if there is only 1 target node and if repeated connections are allowed
            if len(tgt) == 1 and tgt[0].rep and src[0].rep:
                if initialize:
                    self.is_collapsed = True
                return self.is_collapsed

            return False

        # Check all target nodes have 0 or 1 connections
        if any(n.conns != [0, 1] for n in tgt):
            return False

        if initialize:
            self.is_collapsed = False
        return not self.is_collapsed

    def _encode_effective(self, effective_settings: MatrixGenSettings, existence: NodeExistence) -> List[DiscreteDV]:
        if self.is_collapsed:
            src, tgt = effective_settings.src[0], effective_settings.tgt[0]
            min_n_conn = max(src.min_conns if src.max_inf else src.conns[0],
                             tgt.min_conns if tgt.max_inf else tgt.conns[0])
            max_n_conn = effective_settings.get_max_conn_matrix()[0, 0]
            n_opts = max_n_conn-min_n_conn+1
            self._min_max_map[existence] = (min_n_conn, max_n_conn)
        else:
            n_opts = len(effective_settings.tgt)

        # One design variable with n_tgt options
        if n_opts < 2:
            return []
        return [DiscreteDV(n_opts=n_opts, conditionally_active=False)]

    def _decode_effective(self, vector: DesignVector, effective_settings: MatrixGenSettings, existence: NodeExistence) \
            -> Tuple[DesignVector, np.ndarray]:
        # If we are in collapsed mode, there is only one target
        if self.is_collapsed:
            if existence not in self._min_max_map:
                raise RuntimeError(f'Unexpected existence pattern: {existence}')
            min_n_conn, max_n_conn = self._min_max_map[existence]
            if len(vector) == 0:
                n_conns = min_n_conn
            else:
                n_conns = min_n_conn+vector[0]
                if n_conns > max_n_conn:
                    n_conns = max_n_conn
                    vector = [max_n_conn-min_n_conn]

            matrix = np.array([[n_conns]], dtype=int)
            return vector, matrix

        # Determine which target is selected
        n_tgt = len(effective_settings.tgt)
        if len(vector) == 0:
            i_select = 0
        else:
            i_select = vector[0]
            if i_select >= n_tgt:
                i_select = n_tgt
                vector = [i_select]

        # Create 1 x n_tgt matrix
        matrix = np.zeros((1, n_tgt), dtype=int)
        matrix[0, i_select] = 1
        return vector, matrix

    def _do_generate_random_dv_mat(self, n: int, effective_settings: MatrixGenSettings, existence: NodeExistence) \
            -> Tuple[np.ndarray, np.ndarray]:
        if self.is_collapsed:
            if existence not in self._min_max_map:
                raise RuntimeError(f'Unexpected existence pattern: {existence}')
            min_n_conn, max_n_conn = self._min_max_map[existence]

            i_opts = np.arange(max_n_conn-min_n_conn+1)
            design_vectors = np.array([i_opts]).T
            matrices = np.zeros((len(i_opts), 1, 1), dtype=int)
            matrices[:, 0, 0] = min_n_conn+i_opts
            return design_vectors, matrices

        n_tgt = len(effective_settings.tgt)
        design_vectors = np.array([np.arange(n_tgt)]).T
        matrices = np.zeros((n_tgt, 1, n_tgt), dtype=int)
        matrices[:, 0, :] = np.eye(n_tgt, dtype=int)
        return design_vectors, matrices

    def _do_get_all_design_vectors(self, effective_settings: MatrixGenSettings, existence: NodeExistence,
                                   design_vars: List[DiscreteDV]) -> np.ndarray:
        if len(design_vars) == 0:
            return np.zeros((1, 0), dtype=int)
        if self.is_collapsed:
            if existence not in self._min_max_map:
                raise RuntimeError(f'Unexpected existence pattern: {existence}')
            min_n_conn, max_n_conn = self._min_max_map[existence]
            i_opts = np.arange(max_n_conn-min_n_conn+1)
            design_vectors = np.array([i_opts]).T
            return design_vectors

        n_tgt = len(effective_settings.tgt)
        design_vectors = np.array([np.arange(n_tgt)]).T
        return design_vectors

    def _pattern_name(self) -> str:
        return 'Combining'


class AssigningPatternEncoder(PatternEncoderBase):
    """
    Encodes the assigning pattern: connect n_src nodes to n_tgt nodes.
    Optionally surjective: each target node has min 1 connection.
    Optionally repeatable: connections between src and tgt nodes may be repeated.

    Source nodes: n_src nodes with K or more connections (K >= 0)
    Target nodes: n_tgt nodes with any nr of connections (1 or more if surjective)
    Encoded as:   n_src x n_tgt design variables with 2 options each (max_conn_parallel if repeatable)
    """

    def __init__(self, imputer):
        self.surjective = False  # Each tgt min 1 conn
        self.repeatable = False
        self._n_max = 1
        super().__init__(imputer)

    def _matches_pattern(self, effective_settings: MatrixGenSettings, initialize: bool) -> bool:
        if initialize:
            self.surjective = self.repeatable = False
        src, tgt = effective_settings.src, effective_settings.tgt

        # There may be no excluded edges
        if effective_settings.get_excluded_indices():
            return False

        def _set_check(attr, value):
            if initialize:
                setattr(self, attr, value)
                return True

            return getattr(self, attr) == value

        # Check if all repeatability flags are the same
        repeatable = np.array([n.rep for n in src]+[n.rep for n in tgt])
        if not np.all(repeatable) and not np.all(~repeatable):
            return False
        if not _set_check('repeatable', repeatable[0]):
            return False

        # Check if all source nodes have any nr of connections
        if any(not n.max_inf for n in src):
            return False

        # Check if all source nodes have the same minimum nr of connections
        if any(n.min_conns != src[0].min_conns for n in src):
            return False

        # Check if all max parallel connections are the same
        n_max = effective_settings.get_max_conn_parallel() if self.repeatable else 1
        if not _set_check('_n_max', n_max):
            return False

        # Check if surjective (tgt nodes at least 1 conn)
        if all(n.max_inf for n in tgt):
            n_min_conn = np.array([n.min_conns for n in tgt])
            if np.all(n_min_conn == n_min_conn[0]) and n_min_conn[0] in [0, 1]:
                if not _set_check('surjective', n_min_conn[0] == 1):
                    return False
                return True

            return False

        # Do not check for bijective, as that is actually the same as the non-covering partitioning pattern,
        # so there is a better encoder available
        # Do not check for injective, as there the downselecting encoder is better

        return False

    def _encode_effective(self, effective_settings: MatrixGenSettings, existence: NodeExistence) -> List[DiscreteDV]:
        n_max = self._n_max
        return [DiscreteDV(n_opts=n_max+1, conditionally_active=False)
                for _ in range(len(effective_settings.src)*len(effective_settings.tgt))]

    def _decode_effective(self, vector: DesignVector, effective_settings: MatrixGenSettings, existence: NodeExistence) \
            -> Tuple[DesignVector, np.ndarray]:
        surjective = self.surjective
        n_src, n_tgt, n_max = len(effective_settings.src), len(effective_settings.tgt), self._n_max
        n_min_src = effective_settings.src[0].min_conns

        # Correct the specified number of connections
        vector = np.array(vector)
        self._correct_vector(vector, n_src, n_tgt, surjective, n_min_src, n_max,
                             impute_randomly=self._allow_random_imputation)

        # The matrix is simply the reshaped design vector
        matrix = np.array(vector, dtype=int).reshape((n_src, n_tgt))
        return vector, matrix

    @staticmethod
    def _correct_vector(vector: np.ndarray, n_src: int, n_tgt: int, surjective: bool, n_min_src: int, n_max: int,
                        impute_randomly=True):
        for i_tgt in range(n_tgt):
            # Get the connections for the given target node
            conns = vector[i_tgt::n_tgt]

            if surjective and all(conns == 0):
                # If surjective, there should be at least 1 connection per target
                conns[np.random.randint(0, n_src) if impute_randomly else 0] = 1

        if n_min_src > 0:
            for i_src in range(n_src):
                conns = vector[i_src*n_tgt:(i_src+1)*n_tgt]
                n_conns = np.sum(conns)

                while n_conns < n_min_src:
                    i_available = np.where(conns < n_max)[0]
                    i_assign = i_available[np.random.randint(0, len(i_available)) if impute_randomly else 0]
                    conns[i_assign] += 1
                    n_conns += 1

    def _do_generate_random_dv_mat(self, n: int, effective_settings: MatrixGenSettings, existence: NodeExistence) \
            -> Tuple[np.ndarray, np.ndarray]:
        surjective = self.surjective
        n_src, n_tgt, n_max = len(effective_settings.src), len(effective_settings.tgt), self._n_max
        n_min_src = effective_settings.src[0].min_conns

        design_vectors = np.random.randint(0, n_max+1, size=(n, n_src*n_tgt))
        matrices = np.zeros((n, n_src, n_tgt), dtype=int)
        for i in range(design_vectors.shape[0]):
            self._correct_vector(design_vectors[i, :], n_src, n_tgt, surjective, n_min_src, n_max)
            matrices[i, :, :] = design_vectors[i, :].reshape((n_src, n_tgt))

        return design_vectors, matrices

    def _do_get_all_design_vectors(self, effective_settings: MatrixGenSettings, existence: NodeExistence,
                                   design_vars: List[DiscreteDV]) -> np.ndarray:
        surjective = self.surjective
        n_src, n_tgt, n_max = len(effective_settings.src), len(effective_settings.tgt), self._n_max
        n_min_src = effective_settings.src[0].min_conns

        design_vectors = np.array(list(itertools.product(*[list(range(n_max+1)) for _ in range(n_src*n_tgt)])))

        # Remove design vectors with no target connections if surjective
        if surjective:
            for i_tgt in range(n_tgt):
                has_conn = np.any(design_vectors[:, i_tgt::n_tgt] > 0, axis=1)
                design_vectors = design_vectors[has_conn, :]

        # Remove design vectors with not enough source connections
        if n_min_src > 0:
            for i_src in range(n_src):
                has_enough_conn = np.sum(design_vectors[:, i_src*n_tgt:(i_src+1)*n_tgt], axis=1) >= n_min_src
                design_vectors = design_vectors[has_enough_conn, :]

        return design_vectors

    def _pattern_name(self) -> str:
        return 'Assigning'


class PartitioningPatternEncoder(PatternEncoderBase):
    """
    Encodes the partitioning pattern: partition M nodes into max N sets (or N sets of at least size K).
    Also encodes the downselecting pattern: select any from N nodes
    Also encodes the injective assigning pattern: connect M src nodes to N target nodes that accept 0 or 1 connections

    Source nodes: N nodes with K or more connections (K >= 0)
    Target nodes: M nodes with 1 connection (or 0 or 1 if downselecting)
    Encoded as:   M design variables with N options each (or N+1 if downselecting)
    """

    def _matches_pattern(self, effective_settings: MatrixGenSettings, initialize: bool) -> bool:
        src, tgt = effective_settings.src, effective_settings.tgt

        # There may be no excluded edges
        if effective_settings.get_excluded_indices():
            return False

        # Check if all source nodes have any nr of connections
        if any(not n.max_inf for n in src):
            return False

        # Check if all source nodes have the same minimum nr of connections
        if any(n.min_conns != src[0].min_conns for n in src):
            return False

        # Check if all target nodes have 1 connection or 0 or 1 connections
        if any(n.conns != [1] and n.conns != [0, 1] for n in tgt):
            return False

        # Check if there are not too many connections asked for
        n_min_total = src[0].min_conns*len(src)
        if n_min_total > len(tgt):
            return False

        return True

    def _encode_effective(self, effective_settings: MatrixGenSettings, existence: NodeExistence) -> List[DiscreteDV]:
        n_src, n_tgt = len(effective_settings.src), len(effective_settings.tgt)

        # One extra option to also represent the possibility of not selecting any source for a given target
        n_opts = n_src
        if effective_settings.tgt[0].conns == [0, 1]:
            n_opts += 1

        return [DiscreteDV(n_opts=n_opts, conditionally_active=False) for _ in range(n_tgt)]

    def _decode_effective(self, vector: DesignVector, effective_settings: MatrixGenSettings, existence: NodeExistence) \
            -> Tuple[DesignVector, np.ndarray]:
        impute_randomly = self._allow_random_imputation

        # Ensure that each source node has enough connections
        n_src, n_tgt = len(effective_settings.src), len(effective_settings.tgt)
        n_min_src = effective_settings.src[0].min_conns
        tgt_optional = effective_settings.tgt[0].conns == [0, 1]

        if n_min_src > 0:
            # Count the number of connection for each source
            # Note: if tgt connections are optional, the first element here represents "no connection")
            n_conn_src = np.zeros((n_src+1 if tgt_optional else n_src,), dtype=int)
            for i_src in vector:
                n_conn_src[i_src] += 1

            vector = np.array(vector)
            n_conn_diff = n_conn_src-n_min_src
            not_the_first = np.ones((len(n_conn_diff),), dtype=bool)
            if tgt_optional:
                # If target is optional, we don't need to select for the first element, and the number of available is
                # the number of connections (as there is no minimum)
                not_the_first[0] = False
                n_conn_diff[0] = n_conn_src[0]

            while np.any(n_conn_diff < 0):
                # Select the first source with not enough connections
                i_not_enough = np.where((n_conn_diff < 0) & not_the_first)[0][0]

                # Select any of the sources with enough connections (might include the "no connection" element)
                i_enough = np.where(n_conn_diff > 0)[0]
                if len(i_enough) == 0:
                    raise RuntimeError('Not enough to take from!')
                i_take_from = np.random.choice(i_enough) if len(i_enough) > 1 and impute_randomly else i_enough[0]

                # Select any of the elements in the vector that selects the source we take from
                i_tf_in_vector = np.where(vector == i_take_from)[0]
                if len(i_enough) == 0:
                    raise RuntimeError('Not enough values in vector!')
                i_vector_take_from = np.random.choice(i_tf_in_vector) \
                    if len(i_tf_in_vector) > 1 and impute_randomly else i_tf_in_vector[0]

                # Modify the vector and update counts
                vector[i_vector_take_from] = i_not_enough
                n_conn_diff[i_not_enough] += 1
                n_conn_diff[i_take_from] -= 1

        # Set connections to source nodes
        matrix = np.zeros((n_src, n_tgt), dtype=int)
        for i in range(n_tgt):
            i_src = vector[i]
            if tgt_optional:
                if i_src > 0:
                    matrix[i_src-1, i] = 1
            else:
                matrix[i_src, i] = 1

        return vector, matrix

    def _do_generate_random_dv_mat(self, n: int, effective_settings: MatrixGenSettings, existence: NodeExistence) \
            -> Tuple[np.ndarray, np.ndarray]:

        n_src, n_tgt = len(effective_settings.src), len(effective_settings.tgt)
        n_min_src = effective_settings.src[0].min_conns
        tgt_optional = effective_settings.tgt[0].conns == [0, 1]
        offset = 1 if tgt_optional else 0

        design_vectors = np.zeros((n, n_tgt), dtype=int)
        matrices = np.zeros((n, n_src, n_tgt), dtype=int)
        for i in range(n):
            i_tgt_available = np.arange(n_tgt)

            # First ensure that the sources have enough connections
            if n_min_src > 0:
                i_tgt = np.random.permutation(n_tgt)[:n_min_src*n_src]
                for i_src in range(n_src):
                    i_tgt_src = i_tgt[i_src*n_min_src:(i_src+1)*n_min_src]
                    matrices[i, i_src, i_tgt_src] = 1
                    design_vectors[i, i_tgt_src] = i_src+offset
                i_tgt_available = list(set(i_tgt_available)-set(i_tgt))

            # Randomly assign targets to sources
            i_src_conns = np.random.randint(0, n_src+offset, len(i_tgt_available))
            for i_avail, i_tgt in enumerate(i_tgt_available):
                i_src_conn = i_src_conns[i_avail]
                if tgt_optional and i_src_conn == 0:
                    continue
                matrices[i, i_src_conn-offset, i_tgt] = 1
                design_vectors[i, i_tgt] = i_src_conn

        return design_vectors, matrices

    def _do_get_all_design_vectors(self, effective_settings: MatrixGenSettings, existence: NodeExistence,
                                   design_vars: List[DiscreteDV]) -> np.ndarray:
        n_src, n_tgt = len(effective_settings.src), len(effective_settings.tgt)
        n_min_src = effective_settings.src[0].min_conns
        tgt_optional = effective_settings.tgt[0].conns == [0, 1]
        offset = 1 if tgt_optional else 0

        design_vectors = np.array(list(itertools.product(*[list(range(dv.n_opts)) for dv in design_vars])))

        # Remove design vectors where the sources do not have enough connections
        if n_min_src > 0:
            for i_src in range(n_src):
                has_enough_conn = np.sum(design_vectors == i_src+offset, axis=1) >= n_min_src
                design_vectors = design_vectors[has_enough_conn, :]

        return design_vectors

    def _pattern_name(self) -> str:
        return 'Partitioning'


class ConnectingPatternEncoder(PatternEncoderBase):
    """
    Encodes the connecting pattern: connect N src to N tgt nodes, however not to the same indices
    Optionally undirected: only connect one triangle of the matrix (as each connection is seen as bi-directional)

    Source nodes: N nodes with any nr of connections
    Target nodes: N nodes with any nr of connections
    Excluded:     (i, j) if i == j (if directed); (i, j) if i >= j (if undirected)
    Encoded as:   N*(N-1) binary design variables if directed; (N*(N-1))/2 if undirected
    """

    def __init__(self, imputer):
        self.directed = False
        super().__init__(imputer)

    def _matches_pattern(self, effective_settings: MatrixGenSettings, initialize: bool) -> bool:
        src, tgt = effective_settings.src, effective_settings.tgt

        # Check if there are the same nr of src and tgt nodes
        if len(src) != len(tgt):
            return False

        # Check if all nodes have any nr of connections
        if any(n.min_conns != 0 for n in src) or any(n.min_conns != 0 for n in tgt):
            return False

        # Check if all nodes are non-repeatable
        if any(n.rep for n in src) or any(n.rep for n in tgt):
            return False

        # Check if there are any excluded edges
        excluded = set(effective_settings.get_excluded_indices())
        if not excluded:
            return False

        # Check if the diagonal is excluded
        if any((i, i) not in excluded for i in range(len(src))):
            return False

        # Check if nothing from the upper triangle is excluded
        if any((i, j) in excluded for i in range(len(src)) for j in range(len(tgt)) if i < j):
            return False

        # Check if also the lower triangle is excluded
        lower_tri_excluded = np.array([(i, j) in excluded for i in range(len(src)) for j in range(len(tgt)) if i > j])
        no_triangle = len(lower_tri_excluded) == 0
        if no_triangle or (len(lower_tri_excluded) > 0 and np.all(lower_tri_excluded)):
            if initialize:
                self.directed = False
            return (not self.directed) or no_triangle

        # Check if the lower triangle is included
        if np.all(~lower_tri_excluded):
            if initialize:
                self.directed = True
            return self.directed

        return False

    def _encode_effective(self, effective_settings: MatrixGenSettings, existence: NodeExistence) -> List[DiscreteDV]:
        return [DiscreteDV(n_opts=2, conditionally_active=False) for _ in range(self._get_n_dv(effective_settings))]

    def _get_n_dv(self, effective_settings: MatrixGenSettings) -> int:
        # Number of design variables if n*n minus n diagonal connections --> n*(n-1)
        n_src = len(effective_settings.src)
        n_dv = n_src*(n_src-1)

        # If not directed, also the lower triangle cannot be connected two, so we half the nr of design variables
        if not self.directed:
            n_dv = int(n_dv/2)
        return n_dv

    def _decode_effective(self, vector: DesignVector, effective_settings: MatrixGenSettings, existence: NodeExistence) \
            -> Tuple[DesignVector, np.ndarray]:

        # Fill the upper triangle of the matrix
        n_src = len(effective_settings.src)
        matrix = np.zeros((n_src, n_src), dtype=int)
        n_upper = int(len(vector)/2) if self.directed else len(vector)
        matrix[np.triu_indices(n_src, k=1)] = vector[:n_upper]

        # If directed, also fill the lower triangle
        if self.directed:
            matrix[np.tril_indices(n_src, k=-1)] = vector[n_upper:]

        return vector, matrix

    def _do_generate_random_dv_mat(self, n: int, effective_settings: MatrixGenSettings, existence: NodeExistence) \
            -> Tuple[np.ndarray, np.ndarray]:
        n_dv = self._get_n_dv(effective_settings)
        design_vectors = np.random.randint(0, 2, size=(n, n_dv))

        n_src = len(effective_settings.src)
        matrices = np.zeros((n, n_src, n_src), dtype=int)
        n_upper = int(n_dv/2) if self.directed else n_dv
        i_triu = np.triu_indices(n_src, k=1)
        for i in range(n):
            matrices[i, i_triu[0], i_triu[1]] = design_vectors[i, :n_upper]

        if self.directed:
            i_tril = np.tril_indices(n_src, k=-1)
            for i in range(n):
                matrices[i, i_tril[0], i_tril[1]] = design_vectors[i, n_upper:]

        return design_vectors, matrices

    def _do_get_all_design_vectors(self, effective_settings: MatrixGenSettings, existence: NodeExistence,
                                   design_vars: List[DiscreteDV]) -> np.ndarray:
        design_vectors = np.array(list(itertools.product(*[[0, 1] for _ in range(len(design_vars))])))
        return design_vectors

    def _pattern_name(self) -> str:
        return 'Connecting'


class PermutingPatternEncoder(PatternEncoderBase):
    """
    Encodes the permuting pattern: shuffle the order of N elements

    Source nodes: N nodes with 1 connection
    Target nodes: N nodes with 1 connection
    Encoded as:   N-1 design variables with N-i options
    """

    def _matches_pattern(self, effective_settings: MatrixGenSettings, initialize: bool) -> bool:
        src, tgt = effective_settings.src, effective_settings.tgt

        # There may be no excluded edges
        if effective_settings.get_excluded_indices():
            return False

        # Check if there are the same nr of src and tgt nodes
        if len(src) == 1 or len(src) != len(tgt):
            return False

        # Check if all source nodes have 1 connection
        if any(n.conns != [1] for n in src):
            return False

        # Check if all target nodes have 1 connection
        if any(n.conns != [1] for n in tgt):
            return False

        return True

    def _encode_effective(self, effective_settings: MatrixGenSettings, existence: NodeExistence) -> List[DiscreteDV]:
        n = len(effective_settings.src)
        return [DiscreteDV(n_opts=n-i, conditionally_active=False) for i in range(n-1)]

    def _decode_effective(self, vector: DesignVector, effective_settings: MatrixGenSettings, existence: NodeExistence) \
            -> Tuple[DesignVector, np.ndarray]:
        n = len(effective_settings.src)
        matrix = np.zeros((n, n), dtype=int)

        # Set positions: the position of an element depends on the previously selected elements
        matrix[np.arange(n), self._get_abs_pos(vector, n)] = 1
        return vector, matrix

    @staticmethod
    def _get_abs_pos(vector: DesignVector, n: int):
        # Set positions: the position of an element depends on the previously selected elements
        available = list(range(n))
        abs_pos = []
        for i in range(n-1):
            relative_pos = vector[i]
            pos = available[relative_pos]

            abs_pos.append(pos)
            available.remove(pos)

        # Set remaining connection
        abs_pos.append(available[0])
        return abs_pos

    def _do_generate_random_dv_mat(self, n: int, effective_settings: MatrixGenSettings, existence: NodeExistence) \
            -> Tuple[np.ndarray, np.ndarray]:
        n_pos = len(effective_settings.src)
        design_vectors = np.zeros((n, n_pos-1), dtype=int)
        for i_dv in range(design_vectors.shape[1]):
            design_vectors[:, i_dv] = np.random.randint(0, n_pos-i_dv, n)

        matrices = np.zeros((n, n_pos, n_pos), dtype=int)
        i_src = np.arange(n_pos)
        for i, dv in enumerate(design_vectors):
            matrices[i, i_src, self._get_abs_pos(dv, n_pos)] = 1
        return design_vectors, matrices

    def _do_get_all_design_vectors(self, effective_settings: MatrixGenSettings, existence: NodeExistence,
                                   design_vars: List[DiscreteDV]) -> np.ndarray:
        design_vectors = np.array(list(itertools.product(*[list(range(dv.n_opts)) for dv in design_vars])))
        return design_vectors

    def _pattern_name(self) -> str:
        return 'Permuting'


class UnorderedCombiningPatternEncoder(PatternEncoderBase):
    """
    Encodes the unordered combining pattern: take M from N elements, optionally with replacement

    Source nodes: 1 node with M connections
    Target nodes: N nodes with 0 or 1 connection (or any nr of connections if with replacement)
    Encoded as:   N-1 binary design variables (N+M-2 if with replacement)
    """

    def __init__(self, imputer):
        self.with_replacement = False
        super().__init__(imputer)

    def _matches_pattern(self, effective_settings: MatrixGenSettings, initialize: bool) -> bool:
        src, tgt = effective_settings.src, effective_settings.tgt

        # There may be no excluded edges
        if effective_settings.get_excluded_indices():
            return False

        # Check if there is one node with 1 specific amount of connections
        if len(src) != 1 or not src[0].conns or len(src[0].conns) != 1:
            return False

        # Check if all target nodes have 0 or 1 connection
        if all(n.conns == [0, 1] for n in tgt):
            # Check if there are not more connections requested than available
            if src[0].conns[0] > len(tgt):
                return False

            if initialize:
                self.with_replacement = False
            return not self.with_replacement

        # Check if all target nodes have any nr of connections
        if all(n.min_conns == 0 for n in tgt):

            # Check if all connections are repeated
            if all(n.rep for nodes in (src, tgt) for n in nodes):
                if initialize:
                    self.with_replacement = True
                return self.with_replacement

        return False

    def _encode_effective(self, effective_settings: MatrixGenSettings, existence: NodeExistence) -> List[DiscreteDV]:
        n_take = effective_settings.src[0].conns[0]
        n_tgt = len(effective_settings.tgt)
        n_dv = self._get_n_dv(n_take, n_tgt)
        return [DiscreteDV(n_opts=2, conditionally_active=False) for _ in range(n_dv)]

    def _get_n_dv(self, n_take, n_tgt) -> int:
        n_dv = n_tgt-1

        # Special case: if we do not allow replacement and n_tgt is the same as n_take, there is only 1 possibility
        # (namely: take all), and therefore there is no need to choose
        if not self.with_replacement and n_tgt == n_take:
            return 0

        # If we select with replacement, it is the same as if we could select n_take-1 non-replacing positions
        # You can verify:
        # len(list(itertools.combinations_with_replacement(list(range(n_tgt)), n_take)))
        # == len(list(itertools.combinations(list(range(n_tgt+n_take-1)), n_take)))
        if self.with_replacement:
            n_dv += n_take-1
        return n_dv

    def _decode_effective(self, vector: DesignVector, effective_settings: MatrixGenSettings, existence: NodeExistence) \
            -> Tuple[DesignVector, np.ndarray]:
        impute_randomly = self._allow_random_imputation
        n_take = effective_settings.src[0].conns[0]
        n_tgt = len(effective_settings.tgt)

        # Special case
        if not self.with_replacement and n_take == n_tgt:
            return vector, np.ones((1, n_tgt), dtype=int)

        # Ensure we have selected the correct number of elements
        # Note that we have one design variable less than the number of elements that can be selected: the last element
        # is selected if there n_take-1 elements are taken
        vector = np.array(vector)
        n_taken = np.sum(vector)
        if n_taken < n_take-1:
            # Randomly select elements
            i_not_selected = np.where(vector == 0)[0]
            if impute_randomly:
                target = (n_take-1) if np.random.random() > .5 else n_take
                vector[np.random.choice(i_not_selected, target-n_taken, replace=False)] = 1
            else:
                vector[i_not_selected[:n_take-n_taken]] = 1

        elif n_taken > n_take:
            # Randomly deselect elements
            i_selected = np.where(vector == 1)[0]
            if impute_randomly:
                target = (n_take-1) if np.random.random() > .5 else n_take
                vector[np.random.choice(i_selected, n_taken-target, replace=False)] = 0
            else:
                vector[i_selected[:n_taken-n_take]] = 0

        # Select last element if nr of taken elements is n_take-1
        input_vector = vector
        vector = np.concatenate([vector, [0]])
        if np.sum(vector) == n_take-1:
            vector[-1] = 1

        # If with replacement, we have now selected the non-replacing version, so we have to compress
        # For example: [0 0 1 1 0 1] --> [0 0 2 1]
        if self.with_replacement:
            i_selected = np.where(vector == 1)[0]
            i_selected -= np.arange(len(i_selected))

            vector = np.zeros((n_tgt,), dtype=int)
            for i in i_selected:
                vector[i] += 1

        # Set matrix connections in the first row
        matrix = np.array([vector], dtype=int)
        return input_vector, matrix

    def _do_generate_random_dv_mat(self, n: int, effective_settings: MatrixGenSettings, existence: NodeExistence) \
            -> Tuple[np.ndarray, np.ndarray]:
        # Generate all combinations and randomly take from it
        n_take = effective_settings.src[0].conns[0]
        n_tgt = len(effective_settings.tgt)
        val = list(range(n_tgt))
        wr = self.with_replacement
        combinations = list(itertools.combinations_with_replacement(val, n_take)
                            if wr else itertools.combinations(val, n_take))

        if len(combinations) > n:
            i_select = np.random.choice(len(combinations), n, replace=False)
            combinations = [combinations[i] for i in i_select]

        # Encode as design vectors and matrices
        n_dv = self._get_n_dv(n_take, n_tgt)
        design_vectors = np.zeros((len(combinations), n_dv), dtype=int)
        matrices = np.zeros((len(combinations), 1, n_tgt), dtype=int)
        for i, i_selected in enumerate(combinations):
            for i_sel in i_selected:
                matrices[i, 0, i_sel] += 1

            # If with replacement, we have to decompress the selected indices
            if wr:
                i_selected = np.array(i_selected) + np.arange(len(i_selected))

            for i_sel in i_selected:
                # Never select the last element
                if i_sel < n_dv:
                    design_vectors[i, i_sel] = 1

        return design_vectors, matrices

    def _do_get_all_design_vectors(self, effective_settings: MatrixGenSettings, existence: NodeExistence,
                                   design_vars: List[DiscreteDV]) -> np.ndarray:
        n_take = effective_settings.src[0].conns[0]
        n_dv = len(design_vars)

        design_vectors = []
        for n_target in [n_take, n_take-1]:
            for i_selected in itertools.combinations(list(range(n_dv)), n_target):
                vector = np.zeros((n_dv,), dtype=int)
                if len(i_selected) > 0:
                    vector[list(i_selected)] = 1
                design_vectors.append(vector)

        if len(design_vectors) == 0:
            return -np.ones((1, n_dv), dtype=int)
        return np.array(design_vectors)

    def _pattern_name(self) -> str:
        return 'Unordered Combining'
