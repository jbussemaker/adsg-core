import numpy as np
from typing import *
from adsg_core.optimization.assign_enc.matrix import *
from adsg_core.optimization.assign_enc.lazy_encoding import *
from adsg_core.optimization.assign_enc.eager.encodings.grouped_base import GroupedEncoder
from adsg_core.optimization.assign_enc.eager.encodings.group_element import ConnIdxGroupedEncoder

__all__ = ['LazyConnIdxMatrixEncoder', 'ConnCombsEncoder', 'FlatConnCombsEncoder', 'GroupedConnCombsEncoder',
           'ConnIdxCombsEncoder']


class ConnCombsEncoder:
    """Base class for an encoder that encodes a specific set of connections coming from a particular node"""

    def __init__(self, binary=False):
        self.binary = binary
        self._registry = {}

    def encode_prepare(self):
        self._registry = {}

    def encode(self, key, matrix: np.ndarray, n_declared_other_dvs=None) -> List[DiscreteDV]:
        if matrix.shape[0] == 0:
            self._registry[key] = ({}, matrix)
            return []

        dv_grouping_values = self._get_dv_grouping_values(matrix)
        if dv_grouping_values is None:
            self._registry[key] = ({}, matrix)
            return []

        dv_values = GroupedEncoder.group_by_values(
            dv_grouping_values, ordinal_base=2 if self.binary else None, n_declared_start=n_declared_other_dvs)
        self._registry[key] = (self.get_dv_map_for_lookup(dv_values), matrix)

        cond_active = np.any(dv_values == X_INACTIVE_VALUE, axis=0)
        return [DiscreteDV(n_opts=n+1, conditionally_active=cond_active[i])
                for i, n in enumerate(np.max(dv_values, axis=0))]

    @staticmethod
    def get_dv_map_for_lookup(dv_values):
        active_mask_dv_map = {}
        for i, dv in enumerate(dv_values):
            active_mask = np.array(dv != X_INACTIVE_VALUE)
            active_mask_key = tuple(active_mask)
            if active_mask_key not in active_mask_dv_map:
                active_mask_dv_map[active_mask_key] = {}

            active_mask_dv_map[active_mask_key][hash(tuple(dv[active_mask]))] = (i, dv)

        return [(np.array(active_mask, dtype=bool), dv_map) for active_mask, dv_map in active_mask_dv_map.items()]

    @staticmethod
    def lookup_dv(active_mask_dv_map, dv: np.ndarray) -> Optional[Tuple[int, DesignVector]]:
        for active_mask, dv_map in active_mask_dv_map:
            dv_active = hash(tuple(dv[active_mask]))
            idx_and_dv = dv_map.get(dv_active)
            if idx_and_dv is not None:
                return idx_and_dv

    def decode(self, key, vector: np.ndarray) -> Optional[Tuple[DesignVector, np.ndarray]]:
        dv_map, matrix = self._registry[key]
        if matrix.shape[0] == 0:
            return vector, np.zeros((matrix.shape[1],), dtype=np.int64)
        if len(vector) == 0:
            return vector, matrix[0, :]

        idx_and_dv = self.lookup_dv(dv_map, vector)
        if idx_and_dv is not None:
            i, dv = idx_and_dv
            return dv, matrix[i, :]

    def __repr__(self):
        return f'{self.__class__.__name__}(binary={self.binary})'

    def __str__(self):
        name = self._get_name()
        if self.binary:
            name += ' Bin'
        return name

    def _get_dv_grouping_values(self, matrix: np.ndarray) -> Optional[np.ndarray]:
        raise NotImplementedError

    def _get_name(self) -> str:
        raise NotImplementedError


class FlatConnCombsEncoder(ConnCombsEncoder):
    """Encode by counting the nr of possible matrices"""

    def _get_dv_grouping_values(self, matrix: np.ndarray) -> Optional[np.ndarray]:
        return np.array([np.arange(0, matrix.shape[0])]).T

    def _get_name(self) -> str:
        return 'Flat'


class GroupedConnCombsEncoder(ConnCombsEncoder):
    """Encode by grouping the values of the connection matrix"""

    def _get_dv_grouping_values(self, matrix: np.ndarray) -> Optional[np.ndarray]:
        return matrix

    def _get_name(self) -> str:
        return 'Grouped'


class ConnIdxCombsEncoder(GroupedConnCombsEncoder):
    """Encode by grouping the indices of connections in the connection matrix"""

    def _get_dv_grouping_values(self, matrix: np.ndarray) -> Optional[np.ndarray]:
        return ConnIdxGroupedEncoder.get_conn_indices(matrix)

    def _get_name(self) -> str:
        return 'ConnIdx'


class LazyConnIdxMatrixEncoder(LazyEncoder):
    """
    Encodes matrices by the location index of the connection, per source or target connector.
    """

    def __init__(self, imputer: LazyImputer, conn_encoder: ConnCombsEncoder, by_src=True, amount_first=False):
        self._conn_enc = conn_encoder
        self._by_src = by_src
        self._amount_first = amount_first
        super().__init__(imputer)

    def _encode_prepare(self):
        self._dv_idx_map = {}
        self._cache = {}
        self._conn_enc.encode_prepare()

    def _encode(self, existence: NodeExistence) -> List[DiscreteDV]:
        matrix_gen = self._matrix_gen
        max_conn_mat = matrix_gen.get_max_conn_mat(existence)

        # We can either encode connection from each source node or to each target node
        if self._by_src:
            from_nodes, to_nodes = self.src, self.tgt
            has_from, has_to = existence.src_exists_mask(len(self.src)), existence.tgt_exists_mask(len(self.tgt))
            from_n_conn_override, to_n_conn_override = existence.src_n_conn_override, existence.tgt_n_conn_override
            max_from_conn = matrix_gen.get_max_src_appear(existence)
        else:
            from_nodes, to_nodes = self.tgt, self.src
            has_from, has_to = existence.tgt_exists_mask(len(self.tgt)), existence.src_exists_mask(len(self.src))
            from_n_conn_override, to_n_conn_override = existence.tgt_n_conn_override, existence.src_n_conn_override
            max_from_conn = matrix_gen.get_max_tgt_appear(existence)
            max_conn_mat = max_conn_mat.T

        # Encode each reference node separately
        self._dv_idx_map[existence] = dv_idx_map = {}
        all_dvs = []
        for i, node in enumerate(from_nodes):
            # Check if node exists
            if not has_from[i]:
                continue
            # Get a list of possible nr of outgoing/incoming connections
            n_from_conn = from_n_conn_override.get(i, matrix_gen.get_node_conns(node, max_from_conn[i]))

            # Get maximum nr of connections to each target (source) node for this source (target) node
            n_tgt_conns = max_conn_mat[i, :].copy()
            for j, to_node in enumerate(to_nodes):
                if not has_to[j]:
                    n_tgt_conns[j] = 0

            # Get possible connection matrices
            all_failed = True
            matrix_combs_by_n = []
            for n_from in n_from_conn:
                cache_key = (n_from, tuple(n_tgt_conns))
                if cache_key in self._cache:
                    matrix_combs = self._cache[cache_key]
                else:
                    self._cache[cache_key] = matrix_combs = count_src_to_target(n_from, tuple(n_tgt_conns))

                if matrix_combs.shape[0] > 0:
                    all_failed = False

                matrix_combs_by_n.append(matrix_combs)

            # If for any of the connection sources there is no way to make any connection, this is true for all
            if all_failed:
                return []

            # Encode
            keys_n_dv = []
            dvs = []
            if self._amount_first:
                if len(matrix_combs_by_n) > 1:
                    dvs.append(DiscreteDV(n_opts=len(matrix_combs_by_n), conditionally_active=True))

                n_declared_dvs = self.calc_n_declared_design_points(dvs)
                conn_dvs = []
                for k, matrix in enumerate(matrix_combs_by_n):
                    key = (existence, i, k)
                    conn_dv = self._conn_enc.encode(key, matrix, n_declared_other_dvs=n_declared_dvs)
                    conn_dvs.append(conn_dv)
                    keys_n_dv.append((key, len(conn_dv)))
                dvs += LazyEncoder._merge_design_vars(conn_dvs)

            else:
                key = (existence, i)
                conn_dv = self._conn_enc.encode(
                    key, np.row_stack(matrix_combs_by_n), n_declared_other_dvs=self.calc_n_declared_design_points(dvs))
                keys_n_dv.append((key, len(conn_dv)))
                dvs += conn_dv

            dv_idx_map[i] = (len(all_dvs), len(dvs), keys_n_dv)
            all_dvs += dvs
        return all_dvs

    def _decode(self, vector: DesignVector, existence: NodeExistence) -> Optional[Tuple[DesignVector, np.ndarray]]:
        by_src, amount_first = self._by_src, self._amount_first
        dv_idx_map = self._dv_idx_map.get(existence, {})
        matrix = self.get_empty_matrix()
        if not by_src:
            matrix = matrix.T

        imp_vector = [X_INACTIVE_VALUE]*len(vector)
        vector = np.array(vector, dtype=int)
        for i, (i_dv_start, n_dv, keys) in dv_idx_map.items():
            vector_i = vector[i_dv_start:i_dv_start+n_dv]
            i_key = 0
            if amount_first:
                if len(keys) > 1:
                    i_key = vector_i[0]
                    vector_i = vector_i[1:]
            try:
                key, n_dv_i = keys[i_key]
            except IndexError:
                return
            decoded_data = self._conn_enc.decode(key, vector_i[:n_dv_i])
            if decoded_data is None:
                return
            imp_vector_i, sub_matrix = decoded_data
            imp_vector[i_dv_start:i_dv_start+len(imp_vector_i)] = imp_vector_i
            matrix[i, :] = sub_matrix

        if not by_src:
            matrix = matrix.T
        return imp_vector, matrix

    def __repr__(self):
        return f'{self.__class__.__name__}({self._imputer!r}, {self._conn_enc!r}, ' \
               f'by_src={self._by_src}, amount_first={self._amount_first})'

    def __str__(self):
        return f'Lazy Conn Idx By {"Src" if self._by_src else "Tgt"}{" Amnt" if self._amount_first else ""} + ' \
               f'{self._conn_enc!s} + {self._imputer!s}'
