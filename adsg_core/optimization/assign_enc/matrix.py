import os
import math
import numba
import pickle
import hashlib
import itertools
import functools
import numpy as np
from typing import *
from dataclasses import dataclass
from adsg_core.optimization.assign_enc.cache import get_cache_path

__all__ = ['Node', 'NodeExistence', 'NodeExistencePatterns', 'AggregateAssignmentMatrixGenerator', 'count_n_pool_take',
           'MatrixMap', 'MatrixMapOptional', 'count_src_to_target', 'MatrixGenSettings']


class Node:
    """
    Defines a source or target node with a specified number of possible outgoing/incoming connections:
    either an explicit list or only a lower bound.
    """

    def __init__(self, nr_conn_list: List[int] = None, min_conn: int = None, max_conn: int = math.inf,
                 repeated_allowed: bool = True):

        # Flag whether repeated connections to the same opposite node is allowed
        self.rep = repeated_allowed

        # Determine whether we have a set list of connections or only a lower bound
        conns, min_conns = None, None
        if nr_conn_list is not None:
            conns = nr_conn_list

        elif min_conn is not None:
            if max_conn == math.inf:
                min_conns = min_conn
            else:
                conns = list(range(min_conn, max_conn+1))
        else:
            raise ValueError('Either supply a list or a lower limit')
        self.conns = sorted(conns) if conns is not None else None
        self.min_conns = min_conns

    @property
    def max_inf(self):
        return self.conns is None

    def __str__(self):
        if self.conns is None:
            return f'({self.min_conns}..*)'
        return f'({",".join([str(i) for i in self.conns])})'

    def __repr__(self):
        return f'{self.__class__.__name__}(conns={self.conns!r}, min_conns={self.min_conns}, rep={self.rep})'


class NodeExistence:
    """
    Defines a specific src/tgt node existence pattern.
    """

    def __init__(
            self,
            src_exists: Union[List[bool], np.ndarray] = None,
            tgt_exists: Union[List[bool], np.ndarray] = None,
            src_n_conn_override: Dict[int, List[int]] = None,
            tgt_n_conn_override: Dict[int, List[int]] = None,
            max_src_conn_override: int = None,
            max_tgt_conn_override: int = None,
    ):
        self.src_n_conn_override = self._get_n_conn_override(src_exists, src_n_conn_override)
        self.tgt_n_conn_override = self._get_n_conn_override(tgt_exists, tgt_n_conn_override)
        self.max_src_conn_override = max_src_conn_override
        self.max_tgt_conn_override = max_tgt_conn_override
        self._hash = None

        self._src_exists_mask = None
        self._tgt_exists_mask = None
        self._none_exists = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_hash'] = None
        return state

    def __setstate__(self, state):
        state['_hash'] = None
        self.__dict__ = state

    @staticmethod
    def _get_n_conn_override(exists: Union[List[bool], np.ndarray] = None,
                             n_conn_override: Dict[int, List[int]] = None) -> Dict[int, List[int]]:
        if n_conn_override is None and exists is None:
            return {}

        n_conn_override = {} if n_conn_override is None else n_conn_override.copy()
        if exists is not None:
            for i, exist in enumerate(exists):
                if not exist:
                    n_conn_override[i] = [0]
        if len(n_conn_override) == 0:
            return {}
        return n_conn_override

    def has_src(self, i):
        return self.src_n_conn_override.get(i) != [0]

    def has_tgt(self, i):
        return self.tgt_n_conn_override.get(i) != [0]

    def src_exists_mask(self, n_src: int):
        if self._src_exists_mask is not None:
            return self._src_exists_mask

        exist_mask = np.ones((n_src,), dtype=bool)
        for i in range(n_src):
            if not self.has_src(i):
                exist_mask[i] = False
        self._src_exists_mask = exist_mask
        return exist_mask

    def tgt_exists_mask(self, n_tgt: int):
        if self._tgt_exists_mask is not None:
            return self._tgt_exists_mask

        exist_mask = np.ones((n_tgt,), dtype=bool)
        for i in range(n_tgt):
            if not self.has_tgt(i):
                exist_mask[i] = False
        self._tgt_exists_mask = exist_mask
        return exist_mask

    def none_exists(self, n_src: int, n_tgt: int):
        def _none_exists():
            if np.any(self.src_exists_mask(n_src)):
                return False
            if np.any(self.tgt_exists_mask(n_tgt)):
                return False
            return True

        if self._none_exists is None:
            self._none_exists = _none_exists()
        return self._none_exists

    def get_effective_settings(self, settings: 'MatrixGenSettings') \
            -> Tuple['MatrixGenSettings', Dict[int, int], Dict[int, int]]:
        """Returns connection settings with all connection degree overrides of this existence pattern applied"""

        def _get_effective_nodes(nodes: List[Node], n_conn_override: Dict[int, List[int]] = None,
                                 max_conn_override: int = None) -> Tuple[List[Node], Dict[int, int]]:
            effective_nodes = []
            idx_map = {}
            for i, node in enumerate(nodes):
                # Get effective number of connections
                n_conns, min_conn, max_conn = None, None, math.inf
                if i in n_conn_override:
                    n_conns = n_conn_override[i]

                elif max_conn_override is not None:
                    if node.max_inf:
                        min_conn, max_conn = node.min_conns, max_conn_override
                    else:
                        n_conns = [n for n in node.conns if n <= max_conn_override]

                else:
                    n_conns, min_conn = node.conns, node.min_conns

                # Create effective node and check if it is not inactive
                effective_node = Node(n_conns, min_conn=min_conn, max_conn=max_conn, repeated_allowed=node.rep)
                if not effective_node.max_inf and (len(effective_node.conns) == 0 or effective_node.conns == [0]):
                    continue

                idx_map[i] = len(effective_nodes)
                effective_nodes.append(effective_node)

            return effective_nodes, idx_map

        src, src_idx_map = _get_effective_nodes(settings.src, self.src_n_conn_override, self.max_src_conn_override)
        tgt, tgt_idx_map = _get_effective_nodes(settings.tgt, self.tgt_n_conn_override, self.max_tgt_conn_override)

        # Automatically convert to bounded connection nrs if needed
        # For this we need to fix the maximum nr of connections before the conversion
        max_conn_par = MatrixGenSettings(
            src=src, tgt=tgt, max_conn_parallel=settings.max_conn_parallel).get_max_conn_parallel()

        def _get_max_outgoing_conn(ref_nodes: List[Node]):
            n_max_outgoing = 0
            for n in ref_nodes:
                if not n.rep:
                    n_max_outgoing += 1
                elif n.conns is not None:
                    n_max_outgoing += min(max_conn_par, n.conns[-1])
                else:
                    return -1  # If there is a node with no upper bound, we cannot reliably check the max nr of conns
            return n_max_outgoing

        for nodes, n_conn_max in [(tgt, _get_max_outgoing_conn(src)),
                                  (src, _get_max_outgoing_conn(tgt))]:
            if n_conn_max < 1:
                continue
            n_conn_max = max(2, n_conn_max)
            for node in nodes:
                if node.conns is not None:
                    for min_conns in range(n_conn_max):
                        consecutive_n_conns = list(range(min_conns, n_conn_max+1))
                        if node.conns[:len(consecutive_n_conns)] == consecutive_n_conns:
                            # print(f'CONVERTING {node.conns} -> min {min_conns}; {node!r}')
                            node.min_conns = min_conns
                            node.conns = None
                            break

        # Modify excluded edges
        excluded = []
        for i_src, i_tgt in settings.get_excluded_indices():
            # If source or target node are disabled, skip
            if i_src not in src_idx_map or i_tgt not in tgt_idx_map:
                continue

            excluded.append((src_idx_map[i_src], tgt_idx_map[i_tgt]))

        effective_settings = MatrixGenSettings(
            src=src, tgt=tgt, excluded=excluded, max_conn_parallel=max_conn_par)
        return effective_settings, src_idx_map, tgt_idx_map

    def get_transpose(self) -> 'NodeExistence':
        return self.__class__(
            src_n_conn_override=self.tgt_n_conn_override, tgt_n_conn_override=self.src_n_conn_override,
            max_src_conn_override=self.max_tgt_conn_override, max_tgt_conn_override=self.max_src_conn_override)

    def __hash__(self):
        if self._hash is None:
            self._hash = hash((
                tuple([(k, tuple(v)) for k, v in self.src_n_conn_override.items()]) if self.src_n_conn_override is not None else -1,
                tuple([(k, tuple(v)) for k, v in self.tgt_n_conn_override.items()]) if self.tgt_n_conn_override is not None else -1,
                self.max_src_conn_override, self.max_tgt_conn_override,
            ))
        return self._hash

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        src_conn_str = '' if self.src_n_conn_override is None else ';'.join([f'{k}:{",".join([str(vv) for vv in v])}' for k, v in self.src_n_conn_override.items()])
        tgt_conn_str = '' if self.tgt_n_conn_override is None else ';'.join([f'{k}:{",".join([str(vv) for vv in v])}' for k, v in self.tgt_n_conn_override.items()])
        max_src_conn_str = f' ms{self.max_src_conn_override}' if self.max_src_conn_override is not None else ''
        max_tgt_conn_str = f' mt{self.max_tgt_conn_override}' if self.max_tgt_conn_override is not None else ''
        return f'{src_conn_str} _ {tgt_conn_str}{max_src_conn_str}{max_tgt_conn_str}'


@dataclass(frozen=False)
class NodeExistencePatterns:
    patterns: List[NodeExistence]

    def __post_init__(self):
        if len(set(self.patterns)) != len(self.patterns):
            raise RuntimeError('Duplicate node existence patterns')

    @classmethod
    def always_exists(cls) -> 'NodeExistencePatterns':
        return cls([NodeExistence()])

    @classmethod
    def get_all_combinations(cls, src_is_conditional: List[bool], tgt_is_conditional: List[bool]):
        """
        Gets patterns for all combination of conditional existence, e.g.:
        is_conditional = [False, True, True]
        exists = [True, False, False]
        exists = [True, True , False]
        exists = [True, False, True ]
        exists = [True, True , True ]
        """
        src_exist_opts = [[True, False] if is_conditional else [True] for is_conditional in src_is_conditional]
        tgt_exist_opts = [[True, False] if is_conditional else [True] for is_conditional in tgt_is_conditional]

        patterns = []
        for src_exists in itertools.product(*src_exist_opts):
            for tgt_exists in itertools.product(*tgt_exist_opts):
                patterns.append(NodeExistence(src_exists=list(src_exists), tgt_exists=list(tgt_exists)))
        return cls(patterns)

    @classmethod
    def get_increasing(cls, src_is_conditional: List[bool], tgt_is_conditional: List[bool]):
        """
        Gets patterns for conditional existence combinations where node existence is monotonically increasing, e.g.:
        is_conditional = [False, True, True]
        exists = [True, False, False]
        exists = [True, True , False]
        exists = [True, True , True ]
        """

        def _get_exist_opts(is_conditional: List[bool]) -> List[List[bool]]:
            n_cond = sum([is_cond for is_cond in is_conditional])
            exist_opts = []
            for n_cond_exists in range(n_cond+1):
                n_exists = 0
                exist_opt = []
                for is_cond in is_conditional:
                    if is_cond:
                        exist_opt.append(n_exists < n_cond_exists)
                        n_exists += 1
                    else:
                        exist_opt.append(True)
                exist_opts.append(exist_opt)
            return exist_opts

        src_exist_opts = _get_exist_opts(src_is_conditional)
        tgt_exist_opts = _get_exist_opts(tgt_is_conditional)
        return cls([NodeExistence(src_exists=list(src_exists), tgt_exists=list(tgt_exists))
                    for src_exists, tgt_exists in itertools.product(src_exist_opts, tgt_exist_opts)])

    def src_is_conditional(self, i):
        for pattern in self.patterns:
            if not pattern.has_src(i):
                return True
        return False

    def tgt_is_conditional(self, i):
        for pattern in self.patterns:
            if not pattern.has_tgt(i):
                return True
        return False


@dataclass
class MatrixGenSettings:
    src: List[Node]
    tgt: List[Node]
    excluded: List[Union[Tuple[Node, Node], Tuple[int, int]]] = None
    existence: NodeExistencePatterns = None
    max_conn_parallel: int = None

    def get_max_conn_parallel(self):
        """If not given, calculate it by taking the nr such that each target or source can receive at least the minimum
        nr of connections, but at least 2 per connection"""
        if self.max_conn_parallel is not None:
            return max(1, self.max_conn_parallel)

        max_non_inf = 2
        for nodes in [self.src, self.tgt]:
            for node in nodes:
                if not node.max_inf:
                    max_conn = max(node.conns)
                    if max_non_inf is None or max_conn > max_non_inf:
                        max_non_inf = max_conn

        return max_non_inf

    def get_excluded_indices(self) -> List[Tuple[int, int]]:
        excluded_edges = []
        src_map = {node: i for i, node in enumerate(self.src)}
        tgt_map = {node: i for i, node in enumerate(self.tgt)}
        for src, tgt in (self.excluded or []):
            excluded_edges.append((src_map.get(src, src), tgt_map.get(tgt, tgt)))
        return excluded_edges

    def get_max_conn_matrix(self):
        # Restrict by maximum parallel connections
        max_conns = np.ones((len(self.src), len(self.tgt)), dtype=int)*self.get_max_conn_parallel()

        # Restrict by max fixed connection numbers
        for i_src, node in enumerate(self.src):
            if not node.max_inf:
                max_conns[i_src, :] = np.min([max_conns[i_src, :], [max(node.conns)]*len(self.tgt)], axis=0)
        for i_tgt, node in enumerate(self.tgt):
            if not node.max_inf:
                max_conns[:, i_tgt] = np.min([max_conns[:, i_tgt], [max(node.conns)]*len(self.src)], axis=0)

        # Restrict by non-repeatability into/out of nodes
        for i_src, node in enumerate(self.src):
            if not node.rep:
                max_conns[i_src, max_conns[i_src, :] > 1] = 1
        for i_tgt, node in enumerate(self.tgt):
            if not node.rep:
                max_conns[max_conns[:, i_tgt] > 1, i_tgt] = 1

        # Restrict excluded edges
        for i_src, i_tgt in self.get_excluded_indices():
            max_conns[i_src, i_tgt] = 0

        return max_conns

    def get_effective_settings(self) -> Dict[NodeExistence, Tuple['MatrixGenSettings', Dict[int, int], Dict[int, int]]]:
        """For each node existence pattern, get the effective node connection settings"""
        patterns = self.existence.patterns if self.existence is not None else [NodeExistence()]
        return {existence: existence.get_effective_settings(self) for existence in patterns}

    def expand_effective_matrix(self, matrix: np.ndarray, src_map: Dict[int, int], tgt_map: Dict[int, int]) -> np.ndarray:
        if len(src_map) == len(self.src) and len(tgt_map) == len(self.tgt):
            return matrix

        is_2d = len(matrix.shape) == 2
        if is_2d:
            matrix = np.array([matrix])
        expanded_matrix = np.zeros((matrix.shape[0], len(self.src), len(self.tgt)), dtype=int)
        tgt_indices = list(tgt_map.keys())
        for i_src in range(len(self.src)):
            if i_src in src_map:
                expanded_matrix[:, i_src, tgt_indices] = matrix[:, src_map[i_src], :]

        return expanded_matrix[0, :, :] if is_2d else expanded_matrix

    def get_transpose_settings(self) -> 'MatrixGenSettings':
        tgt, src = self.src, self.tgt
        excluded = [(j, i) for i, j in self.get_excluded_indices()] if self.excluded is not None else None

        existence_patterns = None
        if self.existence is not None:
            existence_patterns = NodeExistencePatterns(patterns=[
                existence.get_transpose() for existence in self.existence.patterns])

        return MatrixGenSettings(src=src, tgt=tgt, excluded=excluded, existence=existence_patterns,
                                 max_conn_parallel=self.max_conn_parallel)

    def get_cache_key(self):
        cache_version = '2'  # Increment this if all existing caches should be invalidated

        src_cache_key = ';'.join([repr(s) for s in self.src])
        tgt_cache_key = ';'.join([repr(t) for t in self.tgt])
        excluded_cache_key = ';'.join([f'{tup[0]:d},{tup[1]:d}' for tup in sorted([ex for ex in self.get_excluded_indices()])]) \
            if self.excluded is not None else ''
        exist_cache_key = ';'.join([str(hash(p)) for p in self.existence.patterns]) if self.existence is not None else ''
        cache_str = '||'.join([src_cache_key, tgt_cache_key, excluded_cache_key, exist_cache_key,
                               str(self.max_conn_parallel), cache_version])
        return hashlib.md5(cache_str.encode('utf-8')).hexdigest()


MatrixMap = Dict[NodeExistence, np.ndarray]
MatrixMapOptional = Union[np.ndarray, Dict[NodeExistence, np.ndarray]]


class AggregateAssignmentMatrixGenerator:
    """
    Generator that iterates over all connection possibilities between two sets of objects, with various constraints:
    - Specified number or range of connections per object
    - Repeated connections between objects allowed or not
    - List of excluded connections
    """

    def __init__(self, settings: MatrixGenSettings):
        self._settings = settings
        self._exist = None
        self._max_conn_mat = None
        self._max_src_appear = {}
        self._max_tgt_appear = {}
        self._node_settings = None
        self._n_conn_override_map = {}

    def __setstate__(self, state):
        if '_max_conn_mat' in state and isinstance(state['_max_conn_mat'], np.ndarray):
            state['_max_conn_mat'] = None
        self.__dict__.update(**state)

    @classmethod
    def create(cls, src: List[Node], tgt: List[Node], **kwargs):
        return cls(MatrixGenSettings(src, tgt, **kwargs))

    @property
    def settings(self) -> MatrixGenSettings:
        return self._settings

    @property
    def src(self):
        return self._settings.src

    @property
    def tgt(self):
        return self._settings.tgt

    @property
    def ex(self) -> Optional[List[Tuple[int, int]]]:
        if self._settings.existence is not None:
            return self._settings.get_excluded_indices()

    @property
    def existence_patterns(self) -> NodeExistencePatterns:
        if self._exist is None:
            self._exist = self._settings.existence or NodeExistencePatterns.always_exists()
        return self._exist

    @property
    def max_conn_mat(self) -> Dict[NodeExistence, np.ndarray]:
        if self._max_conn_mat is None:
            self._max_conn_mat = max_conn_mat_map = {}
            for existence, (settings, src_map, tgt_map) in self._settings.get_effective_settings().items():
                max_conn_mat_map[existence] = \
                    self._settings.expand_effective_matrix(settings.get_max_conn_matrix(), src_map, tgt_map)
        return self._max_conn_mat

    def get_max_conn_mat(self, existence: NodeExistence) -> np.ndarray:
        if existence not in self.max_conn_mat:
            return self.max_conn_mat[NodeExistence()]
        return self.max_conn_mat[existence]

    def get_max_src_appear(self, existence: NodeExistence = None):
        if existence is None:
            existence = NodeExistence()
        if existence not in self._max_src_appear:
            max_appear = np.sum(self.get_max_conn_mat(existence), axis=1)
            if existence.max_src_conn_override is not None:
                max_appear[max_appear > existence.max_src_conn_override] = existence.max_src_conn_override
            self._max_src_appear[existence] = max_appear
        return self._max_src_appear[existence]

    def get_max_tgt_appear(self, existence: NodeExistence = None):
        if existence is None:
            existence = NodeExistence()
        if existence not in self._max_tgt_appear:
            max_appear = np.sum(self.get_max_conn_mat(existence), axis=0)
            if existence.max_tgt_conn_override is not None:
                max_appear[max_appear > existence.max_tgt_conn_override] = existence.max_tgt_conn_override
            self._max_tgt_appear[existence] = max_appear
        return self._max_tgt_appear[existence]

    def __iter__(self) -> Generator[Tuple[np.array, NodeExistence], None, None]:
        for n_src_conn, n_tgt_conn, existence in self.iter_n_sources_targets():
            yield self._iter_matrices(n_src_conn, n_tgt_conn, existence), existence  # Iterate over assignment matrices

    def iter_matrices(self, existence: NodeExistence = None):
        for n_src_conn, n_tgt_conn, exist in self.iter_n_sources_targets(existence=existence):
            for matrix in self._iter_matrices(n_src_conn, n_tgt_conn, existence=exist):
                yield matrix, exist

    def get_agg_matrix(self, cache=False) -> MatrixMap:
        if cache:
            agg_matrix = self._load_agg_matrix_from_cache()
            if agg_matrix is not None:
                return agg_matrix

        agg_matrix = self._agg_matrices(self, ensure_all_existence=True)
        self._write_to_cache(self._get_cache_file(), agg_matrix)
        return agg_matrix

    def _load_agg_matrix_from_cache(self) -> Optional[MatrixMap]:
        return self._load_from_cache(self._get_cache_file())

    @staticmethod
    def _write_to_cache(cache_path, obj):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as fp:
            pickle.dump(obj, fp)

    @staticmethod
    def _load_from_cache(cache_path):
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as fp:
                return pickle.load(fp)

    def reset_agg_matrix_cache(self):
        cache_path = self._get_cache_file()
        if os.path.exists(cache_path):
            os.remove(cache_path)

        cache_path = self._get_cache_file_iter_n()
        if os.path.exists(cache_path):
            os.remove(cache_path)

    def _get_cache_file(self):
        return self._cache_path(f'{self._get_cache_key()}.pkl')

    def _get_cache_file_iter_n(self):
        return self._cache_path(f'{self._get_cache_key()}_iter_n.pkl')

    def _get_cache_key(self):
        return self.settings.get_cache_key()

    @staticmethod
    def _cache_path(sub_path=None):
        sel_cache_folder = 'matrix_cache'
        sub_path = os.path.join(sel_cache_folder, sub_path) if sub_path is not None else sel_cache_folder
        return get_cache_path(sub_path=sub_path)

    def _agg_matrices(self, matrix_gen, ensure_all_existence=False):
        """Aggregate generated matrices into one big matrix, one per existence pattern"""
        matrices_by_existence = {}
        for matrix_data in list(matrix_gen):
            if isinstance(matrix_data, np.ndarray):
                matrices, existence = matrix_data, NodeExistence()
            else:
                matrices, existence = matrix_data

            if existence not in matrices_by_existence:
                matrices_by_existence[existence] = []
            matrices_by_existence[existence].append(matrices)

        if ensure_all_existence:
            for existence in self.iter_existence():
                if existence not in matrices_by_existence:
                    matrices_by_existence[existence] = []

        agg_matrices_by_existence = {}
        for existence, matrices in matrices_by_existence.items():
            if len(matrices) == 0:
                agg_matrices_by_existence[existence] = np.zeros((0, len(self.src), len(self.tgt)), dtype=int)
            elif len(matrices) == 1:
                agg_matrices_by_existence[existence] = matrices[0]
            else:
                agg_matrices_by_existence[existence] = np.row_stack(matrices)
        return agg_matrices_by_existence

    def get_matrix_for_existence(self, matrix: MatrixMapOptional, existence: NodeExistence = None) -> np.ndarray:
        if isinstance(matrix, np.ndarray):
            return matrix
        if existence not in matrix:
            return np.empty((0, len(self.src), len(self.tgt)), dtype=int)
        return matrix[existence]

    def iter_conns(self) -> Generator[List[Tuple[Node, Node]], None, None]:
        """Generate lists of edges from matrices"""
        for matrix, _ in self.iter_matrices():
            yield tuple(self.get_conns(matrix))

    @staticmethod
    def get_conn_idx(matrix: np.ndarray) -> List[Tuple[int, int]]:
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

    def iter_n_sources_targets(self, existence: NodeExistence = None, cache=True):
        cache_path = self._get_cache_file_iter_n()
        if cache:
            iter_data = self._load_from_cache(cache_path)
            if iter_data is not None:
                n_src = len(self.src)
                for exist, np_tup in iter_data:
                    if existence is None or existence == exist:
                        for i in range(np_tup.shape[0]):
                            yield tuple(np_tup[i, :n_src]), tuple(np_tup[i, n_src:]), exist
                return

        tuples = {}
        for n_src_conn, n_tgt_conn, exist in self._iter_n_sources_targets():
            if exist not in tuples:
                tuples[exist] = []
            tuples[exist].append((n_src_conn, n_tgt_conn))
            if existence is None or existence == exist:
                yield n_src_conn, n_tgt_conn, exist

        # Store it as a numpy array (more efficient than storing as tuples)
        tuples_efficient = []
        n_src = len(self.src)
        for exist, n_src_n_tgt in tuples.items():
            np_tup = np.empty((len(n_src_n_tgt), n_src+len(self.tgt)), dtype=np.int64)
            for i, (n_src_conn, n_tgt_conn) in enumerate(n_src_n_tgt):
                np_tup[i, :n_src] = n_src_conn
                np_tup[i, n_src:] = n_tgt_conn
            tuples_efficient.append((exist, np_tup))
        self._write_to_cache(cache_path, tuples_efficient)

    def _iter_n_sources_targets(self, existence: NodeExistence = None):
        for exist in (self.iter_existence() if existence is None else [existence]):
            for n_src_conn in self.iter_sources(existence=exist):
                for n_tgt_conn in self.iter_targets(n_source=sum(n_src_conn), existence=exist):
                    yield n_src_conn, n_tgt_conn, exist

    def iter_existence(self, include_none=False):
        yield from self.existence_patterns.patterns
        if include_none:
            yield NodeExistence()

    def iter_sources(self, existence: NodeExistence = None):
        n_conn_override = existence.src_n_conn_override if existence is not None else None
        max_conn = self.get_max_src_appear(existence) if existence is not None else None
        yield from self._iter_conn_slots(self.src, n_conn_override=n_conn_override, max_conn=max_conn)

    def iter_targets(self, n_source, existence: NodeExistence = None):
        n_conn_override = existence.tgt_n_conn_override if existence is not None else None
        max_conn = self.get_max_tgt_appear(existence) if existence is not None else None
        yield from self._iter_conn_slots(
            self.tgt, is_src=False, n=n_source, n_conn_override=n_conn_override, max_conn=max_conn)

    def _iter_conn_slots(self, nodes: List[Node], is_src=True, n=None, n_conn_override: Dict[int, List[int]] = None,
                         max_conn: np.ndarray = None):
        """Iterate over all combinations of number of connections per nodes."""

        if max_conn is None:
            max_conn = self.get_max_src_appear() if is_src else self.get_max_tgt_appear()

        if n_conn_override is None:
            n_conn_override = {}

        # Get all possible number of connections for each node
        n_conns = []
        for i, node in enumerate(nodes):
            if i in n_conn_override:
                n_conns.append(n_conn_override[i])
            else:
                n_conns.append(self.get_node_conns(node, max_conn[i]))

        n_conns = tuple(tuple(n_conn) for n_conn in n_conns)
        return self._iter_conn_slots_inner_(n_conns, n)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _iter_conn_slots_inner_(n_conns, n):
        # If we have an amount constraint, use the matrix generation algorithm
        if n is not None:
            # Check if we have no connection slots available
            if len(n_conns) == 0:
                if n == 0:  # If zero connections are allowed, return an empty solution
                    return [tuple()]
                return []

            n_conn_max = [max(n_conn) if len(n_conn) > 0 else 0 for n_conn in n_conns]
            n_tgt_combs = count_src_to_target(n_src=n, n_src_to_target=tuple(n_conn_max))

            # Filter out invalid amounts
            for i_tgt, n_conn in enumerate(n_conns):
                tgt_n_conns = n_tgt_combs[:, i_tgt]
                n_invalid = set(tgt_n_conns) - set(n_conn)
                if len(n_invalid) > 0:
                    invalid_mask = np.zeros((len(tgt_n_conns),), dtype=bool)
                    for n in n_invalid:
                        invalid_mask |= tgt_n_conns == n
                    n_tgt_combs = n_tgt_combs[~invalid_mask, :]

            conn_slots = []
            for i in reversed(range(n_tgt_combs.shape[0])):  # Reversed for compatibility
                conn_slots.append(tuple(n_tgt_combs[i, :]))
            return conn_slots

        # Iterate over all combinations of number of connections
        conn_slots = []
        for n_conn_nodes in itertools.product(*n_conns):

            n_tot = sum(n_conn_nodes)
            if n is not None and n_tot != n:
                continue

            conn_slots.append(tuple(n_conn_nodes))
        return conn_slots

    def _get_node_settings(self):  # For numba-jit-compiled _check_conns
        def _get_node_settings(nodes: List[Node]):
            n_conns_check = [max(node.conns) for node in nodes if not node.max_inf]
            n_conns_check = 0 if len(n_conns_check) == 0 else max(n_conns_check)
            node_settings = np.zeros((len(nodes), 2+n_conns_check+1), dtype=np.int32)
            for i, node in enumerate(nodes):
                if node.max_inf:
                    node_settings[i, 0] = 1
                    node_settings[i, 1] = node.min_conns
                else:
                    node_settings[i, 2+np.array(node.conns)] = 1
            return node_settings

        if self._node_settings is None:
            self._node_settings = (_get_node_settings(self.src), _get_node_settings(self.tgt))
        return self._node_settings

    def _get_n_conn_override(self, existence: NodeExistence = None):
        if existence is None:
            existence = NodeExistence()

        def _make_n_conn_override(nodes: List[Node], override_map, node_settings):
            if override_map is None or len(override_map) == 0:
                return np.zeros((len(nodes), 0), dtype=np.int8)

            n_override_max = max(max([max(n_conns) for n_conns in override_map.values()]), node_settings.shape[1]-2)
            override_settings = np.zeros((len(nodes), n_override_max+1), dtype=np.int8)
            for i in range(len(nodes)):
                if i in override_map:
                    override_settings[i, np.array(override_map[i])] = 1
                else:
                    override_settings[i, :] = -1
            return override_settings

        if existence not in self._n_conn_override_map:
            src_settings, tgt_settings = self._get_node_settings()
            self._n_conn_override_map[existence] = (_make_n_conn_override(self.src, existence.src_n_conn_override, src_settings),
                                                    _make_n_conn_override(self.tgt, existence.tgt_n_conn_override, tgt_settings))
        return self._n_conn_override_map[existence]

    def validate_matrix(self, matrix: np.ndarray, existence: NodeExistence = None) -> bool:
        """Checks whether a connection matrix is valid"""

        if existence is None:
            existence = NodeExistence()

        src_node_settings, tgt_node_settings = self._get_node_settings()
        src_n_override, tgt_n_override = self._get_n_conn_override(existence=existence)
        max_src = self.get_max_src_appear(existence=existence)
        max_tgt = self.get_max_tgt_appear(existence=existence)
        max_conn_mat = self.get_max_conn_mat(existence)

        return _validate_matrix(matrix, max_conn_mat, src_node_settings,
                                tgt_node_settings, src_n_override, tgt_n_override, max_src, max_tgt)

    @staticmethod
    def _check_conns(n_conns: int, node: Node, max_conn: int) -> bool:
        if n_conns > max_conn:
            return False
        if node.max_inf:
            return node.min_conns <= n_conns
        return n_conns in node.conns

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def get_node_conns(node: Node, max_conn: int):
        if node.max_inf:
            slot_max_conn = max_conn
            return list(range(node.min_conns, slot_max_conn+1))

        return [c for c in node.conns if c <= max_conn]

    def count_all_matrices(self, max_by_existence=True) -> int:
        agg_matrix = self._load_agg_matrix_from_cache()
        if agg_matrix is not None:
            count_by_existence = {existence: matrix.shape[0] for existence, matrix in agg_matrix.items()}
        else:
            count_by_existence = {}
            for n_src_conn, n_tgt_conn, existence in self.iter_n_sources_targets():
                if existence not in count_by_existence:
                    count_by_existence[existence] = 0
                count_by_existence[existence] += self.count_matrices(n_src_conn, n_tgt_conn, existence)

        if max_by_existence:
            return max(count_by_existence.values())
        return sum(count_by_existence.values())

    def get_matrices_by_n_conn(self, n_src_conn, n_tgt_conn, existence: NodeExistence = None):
        return self._iter_matrices(n_src_conn, n_tgt_conn, existence=existence)

    def _iter_matrices(self, n_src_conn, n_tgt_conn, existence: NodeExistence = None):
        """
        Iterate over all permutations of the connections between the source and target objects, taking repetition and
        exclusion constraints into account.

        Inspired by [Selva2017: Patterns in System Architecture Decisions], we are using the assigning pattern and
        representing assignments using a matrix of n_src x n_tgt. Each index (i_src, j_tgt) represents a connection
        between a specific source and target, and the value in the matrix represents the number of connections.

        To create the different matrices, we have to adhere to three constraints:
        1. The sum of target connections (sum over columns) should be equal to the nr of times they occur in tgt_objs
        2. Ensure that in certain rows (sources) or columns (targets), all values are max 1 if no repetition is allowed
        3. Block certain connections from being made at all if we have excluded edges
        """
        if existence is None:
            existence = NodeExistence()

        # Check if we have any source connections
        if len(n_src_conn) == 0 or len(n_tgt_conn) == 0:
            return np.zeros((1, len(n_src_conn), len(n_tgt_conn)), dtype=int)

        # Generate matrices
        return self._get_matrices_numpy(n_src_conn, n_tgt_conn, existence)
        # return self._get_matrices_numba(n_src_conn, n_tgt_conn, existence)

    def count_matrices(self, n_src_conn, n_tgt_conn, existence: NodeExistence = None) -> int:
        """Count the number of permutations as would be generated by _iter_matrices."""
        if existence is None:
            existence = NodeExistence()

        # Count special cases
        n_mat = self._count_matrices_special(n_src_conn, n_tgt_conn, existence)
        if n_mat is not None:
            return n_mat

        # Count symmetric special cases
        n_mat = self._count_matrices_special(n_tgt_conn, n_src_conn, existence, is_switched=True)
        if n_mat is not None:
            return n_mat

        return self._get_matrices_numpy(n_src_conn, n_tgt_conn, existence, create_matrices=False)
        # return len(self._get_matrices_numba(n_src_conn, n_tgt_conn, existence, output_matrix=False))

    def _get_matrices_numba(self, n_src_conn, n_tgt_conn, existence: NodeExistence, output_matrix=True):
        n_src_conn = np.array(n_src_conn, dtype=np.int64)
        n_tgt_conn = np.array(n_tgt_conn, dtype=np.int64)
        max_conn_mat = self.get_max_conn_mat(existence)
        matrices = yield_matrices(
            len(n_src_conn), len(n_tgt_conn), np.array(n_src_conn, dtype=np.int64),
            np.array(n_tgt_conn, dtype=np.int64), max_conn_mat)

        if output_matrix:
            if len(matrices) == 0:
                return np.empty((0, len(n_src_conn), len(n_tgt_conn)), dtype=int)
            elif len(matrices) == 1:
                return np.array([matrices[0]])
            return np.array(matrices)
        return matrices

    def _count_matrices_special(self, n_src_conn, n_tgt_conn, existence: NodeExistence, is_switched=False) \
            -> Optional[int]:
        # No connections
        n_src_total = sum(n_src_conn)
        n_tgt_total = sum(n_tgt_conn)
        if n_src_total == 0 or n_tgt_total == 0:
            return 1

        # 1 outgoing connection
        if len(n_src_conn) == 1 and n_src_conn[0] == 1:
            n_mat = 0
            for n_tgt in n_tgt_conn:
                if n_tgt > 0:
                    n_mat += 1
            return n_mat

        # Get max conn
        max_conn = self.get_max_conn_mat(existence)
        if is_switched:
            max_conn = max_conn.T

        # One of the src nodes has same nr of target connections
        for i_src, n_src in enumerate(n_src_conn):
            if n_src == n_tgt_total:
                # Check if any of the connection nr constraints are violated
                if np.any(n_tgt_conn > max_conn[i_src, :]):
                    return 0
                return 1

    def _get_matrices_numpy(self, n_src_conn, n_tgt_conn, existence: NodeExistence, create_matrices=True):
        max_conn_mat = self.get_max_conn_mat(existence)
        return _get_matrices(
            tuple(n_src_conn), tuple(n_tgt_conn), tuple(max_conn_mat.ravel()), create_matrices=create_matrices)

    def _get_max_conn(self) -> int:
        """
        Calculate the maximum number of connections that will be formed, taking infinite connections into account.
        """

        def _max(nodes):
            if len(nodes) == 0:
                return 0
            return sum([self.get_node_max_conn(node) for node in nodes])

        return max([_max(self.src), _max(self.tgt)])

    @staticmethod
    def get_node_max_conn(node: Node, n_min_max_inf: int = 1):
        return max(n_min_max_inf, node.min_conns) if node.max_inf else max(node.conns)

    @staticmethod
    def _get_min_conns(node: Node) -> int:
        if node.max_inf:
            return node.min_conns
        return node.conns[0]


@numba.njit()
def _validate_matrix(matrix: np.ndarray, max_conn_mat: np.ndarray, src_node_settings: np.ndarray,
                     tgt_node_settings: np.ndarray, src_n_override: np.ndarray, tgt_n_override: np.ndarray,
                     max_src: np.ndarray, max_tgt: np.ndarray) -> bool:
    # Check repeated or blocked connections
    if np.any(matrix > max_conn_mat):
        return False

    # Check source connections
    for i in range(matrix.shape[0]):
        n_src = np.sum(matrix[i, :])
        if n_src < src_n_override.shape[1]:
            override_ok = src_n_override[i, n_src]
            if override_ok == 0:
                return False
            elif override_ok == 1:
                continue
        if not _check_conns(n_src, src_node_settings[i, :], max_src[i]):
            return False

    # Check target connections
    for i in range(matrix.shape[1]):
        n_tgt = np.sum(matrix[:, i])
        if n_tgt < tgt_n_override.shape[1]:
            override_ok = tgt_n_override[i, n_tgt]
            if override_ok == 0:
                return False
            elif override_ok == 1:
                continue
        if not _check_conns(n_tgt, tgt_node_settings[i, :], max_tgt[i]):
            return False

    return True


@numba.njit()
def _check_conns(n_conns, node_settings: np.ndarray, max_conn: int) -> bool:  # See get_node_settings for data structure
    if n_conns > max_conn:
        return False
    if node_settings[0]:  # 0 = max_inf
        return node_settings[1] <= n_conns  # 1 = min_conns

    if n_conns+2 >= len(node_settings):
        return False
    return node_settings[n_conns+2] == 1


@numba.njit()
def yield_matrices(n_src: int, n_tgt: int, n_src_conn, n_tgt_conn, max_conn):
    init_matrix = np.zeros((n_src, n_tgt), dtype=np.int64)
    init_tgt_sum = np.zeros((n_tgt,), dtype=np.int64)
    last_src_idx = n_src-1
    last_tgt_idx = n_tgt-1

    return _branch_matrices(0, init_matrix, init_tgt_sum, n_src_conn, n_tgt_conn, max_conn, n_tgt,
                            last_src_idx, last_tgt_idx)


@numba.njit(parallel=True)
def _branch_matrices(i_src, current_matrix, current_sum, n_src_conn, n_tgt_conn, max_conn, n_tgt,
                     last_src_idx, last_tgt_idx):
    # Get total connections from this source
    n_total_conns = n_src_conn[i_src]

    # Determine the amount of target connections available (per target node) for the current source node
    n_tgt_conn_avbl = n_tgt_conn-current_sum
    for i_tgt in range(len(n_tgt_conn_avbl)):
        if n_tgt_conn_avbl[i_tgt] > max_conn[i_src, i_tgt]:
            n_tgt_conn_avbl[i_tgt] = max_conn[i_src, i_tgt]

    # Get number of minimum connections per target in order to be able to distribute all outgoing connections
    # starting from the first target
    reverse_cum_max_conn = np.cumsum(n_tgt_conn_avbl[::-1])[::-1]
    n_tgt_conn_min = (n_total_conns-reverse_cum_max_conn)[1:]

    # Loop over different source-to-targets connection patterns
    branched_matrices = []
    init_tgt_conns = np.zeros((n_tgt,), dtype=np.int64)
    for tgt_conns in _get_branch_tgt_conns(
            init_tgt_conns, 0, 0, n_total_conns, n_tgt_conn_avbl, n_tgt_conn_min, last_tgt_idx):
        # Modify matrix
        next_matrix = current_matrix.copy()
        next_matrix[i_src, :] = tgt_conns

        # Check if we can stop branching out
        if i_src == last_src_idx:
            branched_matrices.append(next_matrix)
            continue

        # Modify matrix sum
        next_sum = current_sum+tgt_conns

        # Branch out for next src node
        for branched_matrix in _branch_matrices(i_src+1, next_matrix, next_sum, n_src_conn, n_tgt_conn, max_conn,
                                                n_tgt, last_src_idx, last_tgt_idx):
            branched_matrices.append(branched_matrix)

    return branched_matrices


@numba.njit()
def _get_branch_tgt_conns(tgt_conns, i_tgt, n_conn_set, n_total_conns, n_tgt_conn_avbl, n_tgt_conn_min, last_tgt_idx):
    # Check if we already have distributed all connections
    if n_conn_set == n_total_conns:
        return [tgt_conns]
    sub_tgt_conns = []

    # Branch: add another connection to the current target index, if below max for this target
    if tgt_conns[i_tgt] < n_tgt_conn_avbl[i_tgt]:
        next_tgt_conns = tgt_conns.copy()
        next_tgt_conns[i_tgt] += 1
        for sub_tgt_conn in _get_branch_tgt_conns(
                next_tgt_conns, i_tgt, n_conn_set+1, n_total_conns, n_tgt_conn_avbl, n_tgt_conn_min, last_tgt_idx):
            sub_tgt_conns.append(sub_tgt_conn)

    # Branch: move to next target, if we are not at the end and if we are above the minimum connections
    if i_tgt < last_tgt_idx and n_conn_set >= n_tgt_conn_min[i_tgt]:
        for sub_tgt_conn in _get_branch_tgt_conns(
                tgt_conns, i_tgt+1, n_conn_set, n_total_conns, n_tgt_conn_avbl, n_tgt_conn_min, last_tgt_idx):
            sub_tgt_conns.append(sub_tgt_conn)

    return sub_tgt_conns


@functools.lru_cache(maxsize=None)
def _get_matrices(n_src_conn, n_tgt_conn, max_conn_mat_tup, create_matrices=True):
    """Recursive column-wise (target-wise)"""
    n_src, n_tgt = len(n_src_conn), len(n_tgt_conn)
    max_conn_mat = np.array(max_conn_mat_tup).reshape(n_src, n_tgt)

    # Determine number of src connections to first target
    n_target_from_src = np.minimum(n_src_conn, max_conn_mat[:, 0])

    # Get combinations of sources to target assignments
    create_here = create_matrices or len(n_tgt_conn) > 1
    n_taken_combs = count_src_to_target(
        n_tgt_conn[0], tuple(n_target_from_src), create_matrices=create_here)
    if (n_taken_combs.shape[0] if create_here else n_taken_combs) == 0:
        return np.empty((0, n_src, n_tgt), dtype=int) if create_matrices else 0

    # Only one target left to connect to
    if len(n_tgt_conn) == 1:
        if create_matrices:
            return n_taken_combs.reshape((n_taken_combs.shape[0], n_src, 1))
        return n_taken_combs

    # Get combinations for first column
    matrices = []
    count = 0
    n_src_conn_arr = np.array(n_src_conn)
    for i_comb, n_from_src in enumerate(n_taken_combs):
        n_src_conn_remain = n_src_conn_arr-n_from_src
        next_mc_mat = tuple(max_conn_mat[:, 1:].ravel())

        # Get combinations for further columns
        next_matrices = _get_matrices(
            tuple(n_src_conn_remain), n_tgt_conn[1:], next_mc_mat, create_matrices=create_matrices)

        if not create_matrices:
            count += next_matrices
            continue

        matrices_i = np.empty((next_matrices.shape[0], n_src, n_tgt), dtype=int)
        matrices_i[:, :, 0] = n_from_src
        matrices_i[:, :, 1:] = next_matrices
        matrices.append(matrices_i)

    if not create_matrices:
        return count
    return matrices[0] if len(matrices) == 1 else np.row_stack(matrices)


@functools.lru_cache(maxsize=None)
# @numba.njit()
def count_src_to_target(n_src, n_src_to_target, create_matrices=True):
    # Prepare slot duplication definition
    n_flat_slots = np.sum(n_src_to_target)
    slot_i_tgt = np.empty((n_flat_slots,), dtype=np.int64)
    n_dup_next = np.zeros((n_flat_slots,), dtype=np.int64)
    i_slot = 0
    for i_tgt, n in enumerate(n_src_to_target):
        slot_i_tgt[i_slot:i_slot+n] = i_tgt
        if n > 1:
            n_dup_next[i_slot] = n-1
        i_slot += n
    n_dup_next = tuple(n_dup_next)  # For cached version

    # Count combinations for this source node
    count, combinations = count_n_pool_take(len(n_dup_next), n_src, n_dup_next)
    if not create_matrices:
        return count

    # Find nr of connections to slots
    n_taken_combs = np.zeros((count, len(n_src_to_target)), dtype=np.int64)
    for i_tgt, n in enumerate(n_src_to_target):
        if n == 0:
            continue

        conn_comb_tgt = combinations[:, slot_i_tgt == i_tgt]
        n_taken_combs[:, i_tgt] = conn_comb_tgt[:, 0] if n == 1 else np.sum(conn_comb_tgt, axis=1)

    return n_taken_combs


@functools.lru_cache(maxsize=None)
# @numba.njit()
def count_n_pool_take(n_pool, n_take, n_dup_next):
    # Special cases: at the edges of n_take wrt n_pool, we have only 1 combination (duplications don't matter)
    # Example: take 0 from 4 --> 0000; take 4 from 4 --> 1111
    if n_take == 0 or n_take == n_pool:
        combinations = np.zeros((1, n_pool), dtype=np.int64) if n_take == 0 else np.ones((1, n_pool), dtype=np.int64)
        return 1, combinations
    elif n_take < 0 or n_take > n_pool:
        return 0, np.zeros((0, n_pool), dtype=np.int64)

    # We are symmetric wrt n_take, and since the algorithm is a bit faster for lower numbers of n_take, we swap the
    # value if we are higher than half of n_pool
    is_swapped = False
    if n_pool > 1 and n_take > n_pool*.5:
        n_take = n_pool-n_take
        is_swapped = True

    # Get initial combinations matrix
    combinations = _combinations_matrix(n_pool, n_take)

    # Process duplication constraints
    if np.sum(n_dup_next) > 0:
        mask_invalid_dup = np.zeros((combinations.shape[0],), dtype=bool)  # For cached
        # mask_invalid_dup = np.zeros((combinations.shape[0],), dtype=numba.types.bool_)  # For numba
        for i, n_dup in enumerate(n_dup_next):
            if n_dup == 0:
                continue

            # Valid rows have uniformly decreasing or equal number of connections (e.g. 1,1,0, but not 1,0,1 or 0,1,1)
            n_dup_diff = np.diff(combinations[:, i:i+n_dup+1], axis=1)  # For cached
            # n_dup_diff = combinations[:, i+1:i+n_dup+1]-combinations[:, i:i+n_dup]  # For numba
            for j in range(n_dup_diff.shape[1]):
                mask_invalid_dup |= n_dup_diff[:, j] > 0

        # Remove invalid rows
        combinations = combinations[~mask_invalid_dup, :]

    if is_swapped:
        # The consequence of this is that the duplicated slots are filled from the end instead of from the front, but
        # for adding/counting that shouldn't matter
        combinations = (1-combinations)[::-1, :]

    return combinations.shape[0], combinations


@functools.lru_cache(maxsize=None)
# @numba.njit()
def _combinations_matrix(p, r):
    # Based on nump2 of https://stackoverflow.com/a/42202157
    # Prepare combinations matrix and first column
    n0 = p-r+1
    comb_mat = np.zeros((n0, p), dtype=np.int64)
    comb_mat[:n0, :n0] = np.eye(n0)
    n_rep = np.arange(p-r+1)
    # comb_mat[n_rep, n_rep] = 1

    for j in range(1, r):
        # Determine number of repetitions
        reps = (p-r+j) - n_rep
        comb_mat = np.repeat(comb_mat, reps, axis=0)  # For cached
        # comb_mat_rep = np.empty((np.sum(reps), comb_mat.shape[1]), dtype=np.int64)  # For numba
        # i_in_mat = 0
        # for i_rep, n_rep in enumerate(reps):
        #     comb_mat_rep[i_in_mat:i_in_mat+n_rep, :] = comb_mat[i_rep]
        #     i_in_mat += n_rep
        # comb_mat = comb_mat_rep

        ind = np.add.accumulate(reps)  # For cached
        # ind = _add_accumulate(reps)  # For numba

        # Get columns to set to True
        n_rep = np.ones((comb_mat.shape[0],), dtype=np.int64)
        n_rep[ind[:-1]] = 1-reps[1:]
        n_rep[0] = j
        n_rep[:] = np.add.accumulate(n_rep[:])  # For cached
        # n_rep[:] = _add_accumulate(n_rep[:])  # For numba

        comb_mat[np.arange(comb_mat.shape[0]), n_rep] = 1  # For cached
        # for i_rep, i_col in enumerate(n_rep):  # For numba
        #     comb_mat[i_rep, i_col] = True

    return comb_mat


@numba.njit()
def _add_accumulate(arr):
    arr_acc = arr.copy()
    for i in range(1, len(arr)):
        arr_acc[i:] = arr_acc[i-1]+arr[i]
    return arr_acc
