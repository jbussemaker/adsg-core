"""
MIT License

Copyright: (c) 2024, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import math
import copy
import enum
import itertools
import numpy as np
from typing import *
import networkx as nx
from collections import OrderedDict
from adsg_core.graph.graph_edges import *

__all__ = ['ADSGNode', 'ChoiceNode', 'SelectionChoiceNode', 'ConnectionChoiceNode', 'ConnectorNode', 'NamedNode',
           'ConnectorDegreeGroupingNode', 'DesignVariableNode', 'MetricNode', 'MetricType', 'EdgeType', 'EdgeTuple',
           'NodeExportShape']


class NodeExportShape(enum.Enum):
    CIRCLE = 0
    ROUNDED_RECT = 1
    HEXAGON = 2


_CHOICE_COLOR = '#81D4FA'
_INP_OUT_COLOR = '#DCE775'


class ADSGNode:
    """
    Base class for any node used in the ADSG.

    The decision_id attribute is used for grouping nodes into selection-choices to be taken. The option_id attribute
    is used for sorting choice options (if applicable).
    """

    def __init__(self, obj_id=None, decision_id=None, option_id=None, src_decision_link_key=None,
                 tgt_decision_link_key=None, perm_decision_link_key=None, obj_ref=None):
        self._obj_id = obj_id
        self._id = None
        self.update_node_id()
        self.obj_ref = copy.deepcopy(obj_ref)

        self.decision_id = decision_id
        self.option_id = option_id
        self.src_decision_link_key = src_decision_link_key
        self.tgt_decision_link_key = tgt_decision_link_key
        self.perm_decision_link_key = perm_decision_link_key

    def update_node_id(self):
        self._id = hash(self._obj_id or id(self))

    def get_export_shape(self) -> NodeExportShape:
        return NodeExportShape.CIRCLE

    def get_export_color(self) -> str:
        return '#ffffff'

    def get_export_title(self) -> str:
        """Human-readable title for exporting"""
        return str(self)

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        return self._id == hash(other)

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        return str(self)

    def str_context(self):
        """Technical title for exporting"""
        return str(self)

    def copy_node(self, i=None):
        """Copy the node without reference"""
        node_copy = copy.copy(self)
        node_copy.obj_ref = copy.deepcopy(node_copy.obj_ref)
        node_copy.update_node_id()
        return node_copy


class NamedNode(ADSGNode):
    """Generic node with a name"""

    def __init__(self, name: str, **kwargs):
        self.name = name
        super().__init__(**kwargs)

    def get_export_title(self) -> str:
        return self.name

    def __str__(self):
        return f'[{self.name}]'


EdgeTuple = Tuple[ADSGNode, ADSGNode, Hashable, HashableDict]


class ConnectorNode(ADSGNode):
    """
    A node specifying some connection to be made. Specifies data about the number of connections possible using the
    deg_list attribute (specifying a list of allowed number of connections), or using the deg_min and deg_max
    attributes, which specify a range (inclusive). Note that deg_max can also be inf (math.inf), denoting no upper
    limit.
    """

    def __init__(self, name: str = None, deg_spec=None, deg_list=None, deg_min=1, deg_max=None, repeated_allowed=False,
                 **kwargs):
        self.name = name
        if deg_spec is not None:
            deg_list, deg_min, deg_max = self._parse_deg_spec(deg_spec)

        if deg_list is None:
            if deg_min < 0:
                raise ValueError('Min degree should be positive (given: %d)' % deg_min)
            if deg_max is None:
                deg_max = math.inf
            if deg_max < deg_min:
                raise ValueError('Max degree should be equal to or higher than min degree (%d >= %d)' %
                                 (deg_max, deg_min))
            if deg_min == deg_max and deg_min == math.inf:
                raise ValueError('Infinite should be part of a range, not of a single value')
        else:
            deg_min = None
            deg_max = None

        self.deg_list = deg_list
        self.deg_min = deg_min
        self.deg_max = deg_max
        self.repeated_allowed = repeated_allowed

        super(ConnectorNode, self).__init__(**kwargs)

    def is_valid(self, degree):
        """Check whether the given connection degree is valid for the connection constraint"""
        if self.deg_list is not None:
            return degree in self.deg_list
        return self.deg_min <= degree <= self.deg_max

    def update_deg(self, graph: nx.MultiDiGraph, existing_nodes: Set[ADSGNode] = None):
        pass

    @staticmethod
    def _parse_deg_spec(deg_spec):
        """
        Parse various degree specification formats:
        - list of allowed degrees
        - "x..y": range of degrees, where y can also be "inf" for no upper ceiling
        - "*" for zero or more
        - "+" for one or more
        - "?" for zero or one
        - "req": precisely one
        - "opt": zero or one
        - int: precisely the amount

        :param deg_spec:
        :return:
        """

        if isinstance(deg_spec, (list, tuple)):  # Degree list
            return deg_spec, None, None

        if deg_spec in ('?', 'opt'):  # Zero or one
            deg_min, deg_max = 0, 1
        elif deg_spec == '*':  # Zero or more
            deg_min, deg_max = 0, math.inf
        elif deg_spec == '+':  # One or more
            deg_min, deg_max = 1, math.inf
        elif deg_spec == 'req':  # Precisely 1
            return [1], None, None
        else:
            try:
                if '..' in deg_spec:  # Range
                    deg_min, deg_max = deg_spec.split('..')
                    deg_min = int(deg_min)
                    deg_max = math.inf if deg_max in ('inf', '*') else int(deg_max)
                else:
                    return [int(deg_spec)], None, None
            except TypeError:  # A specified amount
                return [int(deg_spec)], None, None
        return None, deg_min, deg_max

    @classmethod
    def get_deg_kwargs(cls, deg_spec):
        """
        Get the degree-related port init kwargs, from various specification formats:
        - list of allowed degrees
        - "x..y": range of degrees, where y can also be "inf" for no upper ceiling
        - "*" for zero or more
        - "+" for one or more
        - "?" for zero or one
        - "req": precisely one
        - "opt": zero or one
        - int: precisely the amount

        :param deg_spec:
        :return:
        """
        deg_list, deg_min, deg_max = cls._parse_deg_spec(deg_spec)
        return {'deg_list': deg_list, 'deg_min': deg_min, 'deg_max': deg_max}

    @property
    def deg_str(self):
        if self.deg_list is not None:
            return ' @ %s' % (','.join([str(deg) for deg in self.deg_list]),)
        elif self.deg_min == self.deg_max:
            return '' if self.deg_min == 1 else (' @ %d' % self.deg_min)
        else:
            return ' @ %s..%s' % (self.deg_min, self.deg_max)

    def get_full_deg_str(self):
        if self.deg_list is not None:
            return ','.join([str(deg) for deg in self.deg_list])
        if self.deg_min == self.deg_max:
            return str(self.deg_min)
        if math.isinf(self.deg_max):
            return f'{self.deg_min}..*'
        return f'{self.deg_min}..{self.deg_max}'

    def get_export_title(self) -> str:
        return self.name

    def get_export_shape(self) -> NodeExportShape:
        return NodeExportShape.HEXAGON

    def __str__(self):
        name_str = f'[{self.name}]' if self.name is not None else ''
        return f'Conn{name_str}{self.deg_str}'


class ConnectorDegreeGroupingNode(ConnectorNode):
    """
    Groups a bunch of connectors. Used when the connection order of the connectors does not matter. This means that this
    class only looks at the aggregate number of connections accepted by the grouped connectors. The update_deg function
    should be called with the grouped nodes to update the connection degree specifications.
    """

    def __init__(self, name: str = None):
        super(ConnectorDegreeGroupingNode, self).__init__(name)

    def __str__(self):
        return 'Grp[Conn]'

    def update_deg(self, graph: nx.MultiDiGraph, existing_nodes: Set[ADSGNode] = None):

        connectors: List[ConnectorNode] = \
            [in_edge[0] for in_edge in iter_in_edges(graph, self, edge_type=EdgeType.DERIVES)]
        if existing_nodes is not None:
            connectors = [node for node in connectors if node in existing_nodes]

        self.deg_list, self.deg_min, self.deg_max = self.get_combined_deg(connectors)
        self.repeated_allowed = self.get_repeated_allowed(connectors)

        # Update decision link key
        for node in connectors:  # type: ADSGNode
            if node.perm_decision_link_key:
                self.perm_decision_link_key = node.perm_decision_link_key
                break

    @staticmethod
    def get_repeated_allowed(connectors: List[ConnectorNode]) -> bool:
        for connector in connectors:
            if connector.repeated_allowed:
                return True
        return False

    @staticmethod
    def get_combined_deg(connectors: List[ConnectorNode]) \
            -> Tuple[Optional[List[int]], Optional[int], Optional[Union[int, float]]]:

        # Gather specifications from connectors
        deg_lists = []
        deg_min_inf = None
        for connector in connectors:
            conn_deg_list = connector.deg_list
            conn_deg_min, conn_deg_max = connector.deg_min, connector.deg_max
            if conn_deg_list is not None:
                deg_lists.append(conn_deg_list)
            else:
                if conn_deg_max != math.inf:
                    deg_lists.append(list(range(conn_deg_min, conn_deg_max+1)))
                else:
                    if deg_min_inf is None:
                        deg_min_inf = 0
                    deg_min_inf += conn_deg_min

        # If there is an infinite upper limit, treat the allowed degrees as a list
        if deg_min_inf is not None:
            deg_min_inf += sum([min(deg_list) for deg_list in deg_lists])
            return None, deg_min_inf, math.inf

        # Get all unique connection combinations
        deg_list = sorted(list({sum(comb) for comb in itertools.product(*deg_lists)}))
        return deg_list, None, None

    def get_export_title(self) -> str:
        return 'Grp'

    def get_export_shape(self) -> NodeExportShape:
        return NodeExportShape.HEXAGON


class DesignVariableNode(ADSGNode):
    """
    Node representing a design variable.
    """

    def __init__(self, name, bounds=None, options=None, idx=None, **kwargs):
        if bounds is not None:
            if len(bounds) != 2:
                raise ValueError('Bounds should be a tuple or list of length 2')
            if bounds[0] >= bounds[1]:
                raise ValueError('Upper bound should be higher than lower bound')
            if options is not None:
                raise ValueError('Either bounds or options should be provided, not both!')

        elif options is not None:
            if len(options) < 1:
                raise ValueError('Options list should contain at least one value')

        self.name = name
        self.bounds = bounds
        self.options = options
        self.idx = idx
        super(DesignVariableNode, self).__init__(**kwargs)

        if self.decision_id is None:
            self.decision_id = f'DV_{self.name}'

    @property
    def is_discrete(self):
        """Whether the design variable is discrete or continuous"""
        return self.options is not None

    def correct_value(self, value):
        """Ensure a value is within the bounds or represents a valid option"""
        if self.bounds is None and self.options is None:
            raise ValueError(f'Design variable bounds or options not set: {self!r}')

        bounds_fraction = .5
        if self.is_discrete:
            if value < 0:
                value = 0
            elif value >= len(self.options):
                value = len(self.options)-1
        else:
            lower, upper = self.bounds
            if value < lower:
                value = lower
            elif value > upper:
                value = upper
            bounds_fraction = (value - lower) / (upper - lower)
        return value, bounds_fraction

    def get_export_title(self) -> str:
        if self.is_discrete:
            values_str = ' ['+','.join([str(opt) for opt in self.options])+']'
        else:
            values_str = f' [{self.bounds[0]} .. {self.bounds[1]}]'
        return f'{self.name} {values_str}'

    def get_export_color(self) -> str:
        return _INP_OUT_COLOR

    def str_context(self):
        return f'DV.{self.name}.{self.idx}.{self.bounds!r}.{self.options!r}'

    def __str__(self):
        return f'DV[{self.name}]'


class MetricType(enum.Flag):
    NONE = 0
    OBJECTIVE = enum.auto()
    CONSTRAINT = enum.auto()
    OBJ_OR_CON = OBJECTIVE | CONSTRAINT


class MetricNode(ADSGNode):
    """
    Node representing a performance metric. In the end, a performance metric can either be used as an
    objective or as a constraint in an optimization problem.

    If a direction (meaning which direction of the value is "better") is given, it can be used an objective. If also a
    reference is given (meaning it should be in a specified direction, but at least this value), it can also be used
    as a constraint.

    Metric type is derived automatically, however to force the type you can supply a `MetricType`:

    - `NONE`: no role in the optimization problem
    - `OBJECTIVE`: metric is an objective (provide a direction)
    - `CONSTRAINT`: metric is a constraint (provide a direction and a reference value)
    """

    def __init__(self, name, direction: int = None, ref: float = None, idx=None, type_=None):
        self.name = name
        self.idx = idx
        self.dir = direction  # -1 for min/lte, 1 for max/gte
        self.ref = ref  # Reference value for constraint
        self.type: Optional[MetricType] = type_
        super(MetricNode, self).__init__()

    def get_export_title(self) -> str:
        role_str = ''
        if self.dir is not None:
            if self.ref is not None:
                role_str = ' [>=' if self.dir > 0 else ' <='
                role_str += f' {self.ref!s}]'
            else:
                role_str = ' [max]' if self.dir > 0 else ' [min]'
        return self.name+role_str

    def get_export_color(self) -> str:
        return _INP_OUT_COLOR

    def get_export_shape(self) -> NodeExportShape:
        return NodeExportShape.ROUNDED_RECT

    def str_context(self):
        return f'MET.{self.name}.{self.idx}.dir{self.dir}.ref{self.ref}'

    def __str__(self):
        return 'PR[%s]' % self.name


class ChoiceNode(ADSGNode):
    """
    Node representing a decision to be taken.
    """

    def __init__(self, decision_id=None, decision_sort_key=None):
        self.decision_sort_key = decision_sort_key
        super(ChoiceNode, self).__init__(decision_id=decision_id)

    def get_export_title(self) -> str:
        return self.decision_id

    def get_export_color(self) -> str:
        return _CHOICE_COLOR

    def __str__(self):
        raise NotImplementedError


class SelectionChoiceNode(ChoiceNode):
    """
    Node representing a selection: a choice with a discrete set of mutually exclusive options to
    choose from.
    """

    def __str__(self):
        return 'D[Sel: %s]' % (self.decision_id or '_')

    def str_context(self):
        return 'D[Sel: %s]' % self.decision_id


class ConnectionChoiceNode(ChoiceNode):
    """
    Node representing a connection choice: a choice about how to connect between a set of source nodes and a set of
    target nodes. Sources and targets should be `ConnectorNode` types.
    """

    def __init__(self, decision_id=None, decision_sort_key=None):
        super(ConnectionChoiceNode, self).__init__(decision_id=decision_id, decision_sort_key=decision_sort_key)

    def __str__(self):
        return 'D[Conn: %s]' % (self.decision_id or '_')

    def str_context(self):
        return 'D[Conn: %s]' % self.decision_id

    def iter_conn_edges(self, adsg, check_conditional=False):
        from adsg_core.optimization.assign_enc.matrix import NodeExistence

        matrix_gen, node_map = self._get_matrix_gen(adsg, check_conditional=check_conditional)
        agg_matrix = matrix_gen.get_agg_matrix(cache=True)[NodeExistence()]
        for i_mat in range(agg_matrix.shape[0]):
            conn_edges = matrix_gen.get_conn_idx(agg_matrix[i_mat, :, :])
            node_edges = [(node_map[0][conn_edge[0]], node_map[1][conn_edge[1]]) for conn_edge in conn_edges]
            yield node_edges

    def validate_conn_edges(self, adsg, edges: Sequence[Tuple[ConnectorNode, ConnectorNode]]) -> bool:
        matrix_gen, node_map = self._get_matrix_gen(adsg)
        src_idx_map = {node: i for i, node in enumerate(node_map[0])}
        tgt_idx_map = {node: i for i, node in enumerate(node_map[1])}

        matrix = np.zeros((len(node_map[0]), len(node_map[1])), dtype=int)
        for src, tgt in edges:
            if src not in src_idx_map or tgt not in tgt_idx_map:
                return False
            matrix[src_idx_map[src], tgt_idx_map[tgt]] += 1

        return matrix_gen.validate_matrix(matrix)

    def _get_matrix_gen(self, adsg, check_conditional=False):
        from adsg_core.optimization.assign_enc.matrix import AggregateAssignmentMatrixGenerator

        settings, node_map = self._get_assign_nodes(adsg, check_conditional=check_conditional)
        matrix_gen = AggregateAssignmentMatrixGenerator(settings)
        return matrix_gen, node_map

    def get_assignment_encoding_args(self, adsg, hierarchy_analyzer=None):
        from adsg_core.optimization.assign_enc.matrix import NodeExistencePatterns, NodeExistence, MatrixGenSettings
        from adsg_core.optimization.hierarchy import HierarchyAnalyzer

        if hierarchy_analyzer is None:
            hierarchy_analyzer = HierarchyAnalyzer(adsg)

        # Create connection nodes
        settings, node_map = self._get_assign_nodes(adsg)

        # Get existence patterns
        flat_idx_map = {}
        conn_nodes = list(node_map[0])+list(node_map[1])
        derivation_nodes = self.get_conn_node_derivations(adsg.graph, conn_nodes)
        all_conn_nodes = []
        for conn_node, deriving_conn_nodes in derivation_nodes.items():
            flat_idx_map[conn_node] = len(all_conn_nodes)
            all_conn_nodes.append(conn_node)

            for deriving_node in deriving_conn_nodes:
                flat_idx_map[deriving_node] = len(all_conn_nodes)
                all_conn_nodes.append(deriving_node)

        def _exist_process(assign_nodes, exists, n_conn_override, sub_node_map, n_max):
            for ii in range(len(assign_nodes)):
                conn_node_ = sub_node_map[ii]

                # Check connector node existence
                if not existence_mask[flat_idx_map[conn_node_]]:
                    exists[ii] = False
                    continue

                # Check connection number override
                if isinstance(conn_node_, ConnectorDegreeGroupingNode) and len(derivation_nodes[conn_node_]) > 0:
                    existing_connectors = [der_conn_node for der_conn_node in derivation_nodes[conn_node_]
                                           if existence_mask[flat_idx_map[der_conn_node]]]
                    deg_list, deg_min, deg_max = conn_node_.get_combined_deg(existing_connectors)
                    if deg_list is None:
                        if deg_max == math.inf:
                            deg_max = n_max[ii]
                        deg_list = list(range(deg_min, (deg_max or deg_min)+1))
                    n_conn_override[ii] = deg_list

        # Get possible connector node existences, if not available, generate the cartesian product
        map_by_existence_mask = False
        existence_masks = hierarchy_analyzer.get_nodes_existence(all_conn_nodes)
        if existence_masks is not None:
            unique_existence_masks, existence_map = np.unique(existence_masks, axis=0, return_inverse=True)

        else:
            map_by_existence_mask = True
            unique_existence_masks = np.array(list(itertools.product(
                *[[False, True] for _ in range(len(all_conn_nodes))])))
            existence_map = np.arange(0, len(unique_existence_masks))

        max_conn_matrix = settings.get_max_conn_matrix()

        patterns = []
        pattern_idx_map = {}
        src_nodes, tgt_nodes = settings.src, settings.tgt
        for i_exist, existence_mask in enumerate(unique_existence_masks):
            src_exists, tgt_exists = [True for _ in range(len(src_nodes))], [True for _ in range(len(tgt_nodes))]
            src_n_conn_override, tgt_n_conn_override = {}, {}

            _exist_process(src_nodes, src_exists, src_n_conn_override, node_map[0], np.sum(max_conn_matrix, axis=1))
            _exist_process(tgt_nodes, tgt_exists, tgt_n_conn_override, node_map[1], np.sum(max_conn_matrix, axis=0))

            existence = NodeExistence(
                src_exists=src_exists, tgt_exists=tgt_exists,
                src_n_conn_override=src_n_conn_override, tgt_n_conn_override=tgt_n_conn_override,
            )

            # Prevent duplicates
            if existence in pattern_idx_map:
                existence_map[existence_map == i_exist] = pattern_idx_map[existence]
                continue

            pattern_idx_map[existence] = i_pattern = len(patterns)
            existence_map[existence_map == i_exist] = i_pattern
            patterns.append(existence)

        existence_patterns = NodeExistencePatterns(patterns=patterns)
        settings = MatrixGenSettings(settings.src, settings.tgt, settings.excluded, existence=existence_patterns)

        # Enable identification by node existence rather than combination idx
        if map_by_existence_mask:
            existence_map = {tuple(unique_existence_masks[i_mask]): i_pattern
                             for i_mask, i_pattern in enumerate(existence_map)}

        return settings, node_map, existence_map, all_conn_nodes

    def _get_assign_nodes(self, adsg, check_conditional=False):
        from adsg_core.optimization.assign_enc.matrix import Node as AssignNode, MatrixGenSettings

        graph = adsg.graph
        src_nodes = self.get_src_nodes(graph)
        tgt_nodes = self.get_tgt_nodes(graph)

        for conn_nodes in [src_nodes, tgt_nodes]:
            for conn_node in conn_nodes:
                if isinstance(conn_node, ConnectorDegreeGroupingNode):
                    conn_node.update_deg(adsg.graph)

        # Check whether any of the nodes exist conditionally
        def _is_conditional(connector_node):
            if adsg.has_conditional_existence(connector_node):
                return True

            # Check ancestor port nodes if this is a port grouping node
            if isinstance(connector_node, ConnectorDegreeGroupingNode):
                for prev_node in graph.predecessors(connector_node):
                    if adsg.has_conditional_existence(prev_node):
                        return True

            return False

        src_conditional = set()
        tgt_conditional = set()
        if check_conditional:
            src_conditional = {node for node in src_nodes if _is_conditional(node)}
            tgt_conditional = {node for node in tgt_nodes if _is_conditional(node)}

        src_obj_map = OrderedDict([(node, self.to_assign_node(node, is_conditional=node in src_conditional))
                                   for node in src_nodes])
        tgt_obj_map = OrderedDict([(node, self.to_assign_node(node, is_conditional=node in tgt_conditional))
                                   for node in tgt_nodes])
        node_map = (src_nodes, tgt_nodes)

        src_objs: List[AssignNode] = list(src_obj_map.values())
        tgt_objs: List[AssignNode] = list(tgt_obj_map.values())

        # Get excluded edges
        excluded = []
        for edge in self.get_excluded_edges(graph):
            if edge[0] not in src_obj_map or edge[1] not in tgt_obj_map:
                continue
            excluded.append((src_obj_map[edge[0]], tgt_obj_map[edge[1]]))

        settings = MatrixGenSettings(src_objs, tgt_objs, excluded)
        return settings, node_map

    def get_src_nodes(self, graph) -> List[ConnectorNode]:
        return self.get_sorted_connector_nodes([node for node in graph.predecessors(self)])

    def get_tgt_nodes(self, graph) -> List[ConnectorNode]:
        return self.get_sorted_connector_nodes([node for node in graph.successors(self)])

    @staticmethod
    def get_conn_node_derivations(graph, nodes: List[ConnectorNode]) \
            -> Dict[ConnectorNode, List[ConnectorNode]]:
        node_derivations = {}
        for node in nodes:
            deriving_nodes = []
            if isinstance(node, ConnectorDegreeGroupingNode):
                for prev_node in graph.predecessors(node):
                    if isinstance(prev_node, ConnectorNode):
                        deriving_nodes.append(prev_node)
            node_derivations[node] = deriving_nodes
        return node_derivations

    @staticmethod
    def get_sorted_connector_nodes(nodes):
        return sorted([node for node in nodes if isinstance(node, ConnectorNode)],
                      key=lambda n: (str(n.option_id or ''), str(n.decision_id or '')))

    def get_excluded_edges(self, graph):
        excluded = []
        for node in self.get_src_nodes(graph):
            excluded += [edge for edge in iter_out_edges(graph, node, edge_type=EdgeType.EXCLUDES)]
        return excluded

    def get_deriving_edges(self, graph):
        tgt_nodes = self.get_tgt_nodes(graph)
        deriving_edges = []
        for node in self.get_src_nodes(graph):
            for edge in iter_out_edges(graph, node):
                if edge[1] in tgt_nodes and get_edge_type(edge) == EdgeType.DERIVES:
                    deriving_edges.append(edge)
        return deriving_edges

    @staticmethod
    def to_assign_node(connector_node: ConnectorNode, is_conditional=False):
        from adsg_core.optimization.assign_enc.matrix import Node as AssignNode

        deg_list = connector_node.deg_list
        deg_min = connector_node.deg_min
        deg_max = connector_node.deg_max
        if deg_max is None:
            deg_max = deg_min

        # If this node exists conditionally, simply add the option to accept 0 connections
        if is_conditional:
            if deg_list is not None:
                deg_list = [0]+list(deg_list)
            else:
                deg_min = 0

        return AssignNode(
            nr_conn_list=deg_list,
            min_conn=deg_min,
            max_conn=deg_max,
            repeated_allowed=connector_node.repeated_allowed,
        )

    def get_export_shape(self) -> NodeExportShape:
        return NodeExportShape.HEXAGON
