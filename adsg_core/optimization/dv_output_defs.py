import enum
import random
from typing import *
from adsg_core.graph.adsg_nodes import MetricNode, DesignVariableNode, ChoiceNode

__all__ = ['DesVar', 'Direction', 'Objective', 'Constraint']


class DesVar:
    """
    Class representing a design variable. A design variable can either be discrete (options are specified) or continuous
    (bounds are specified).
    """

    def __init__(self, name: str, options: list = None, bounds: Tuple[float, float] = None,
                 node: Union[DesignVariableNode, ChoiceNode] = None, conditionally_active=False):
        if (options is None) == (bounds is None):
            raise ValueError('Either options or bounds must be provided: %s' % name)
        if options is not None:
            if len(options) == 0:
                raise ValueError('At least one option should be provided: %s' % name)
        if bounds is not None:
            if len(bounds) != 2:
                raise ValueError('Bounds should be a tuple: %s' % name)
            if bounds[0] >= bounds[1]:
                raise ValueError('Lower bound should be lower than upper bound: %.2f < %.2f (%s)' %
                                 (bounds[0], bounds[1], name))

        self._name = name
        self._opts = options
        self._bounds = bounds
        self._node = node
        self.conditionally_active = conditionally_active

    @classmethod
    def from_des_var_node(cls, des_var_node: DesignVariableNode, conditionally_active=False) -> 'DesVar':
        name = des_var_node.name
        if des_var_node.idx is not None:
            name = '%s_%d' % (name, des_var_node.idx)
        return cls(name, bounds=des_var_node.bounds, options=des_var_node.options, node=des_var_node,
                   conditionally_active=conditionally_active)

    @classmethod
    def from_choice_node(cls, choice_node: ChoiceNode, options: list, name: str = None,
                         existing_names: set = None, conditionally_active=False) -> 'DesVar':
        if name is None:
            name = choice_node.decision_id

        if existing_names is not None:
            name_base = name
            i = 2
            while name in existing_names:
                name = f'{name_base}_{i}'
                i += 1

        return cls(name, options=options, node=choice_node, conditionally_active=conditionally_active)

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_discrete(self) -> bool:
        return self._opts is not None

    @property
    def options(self) -> Optional[list]:
        return self._opts

    @property
    def n_opts(self) -> Optional[int]:
        return len(self._opts) if self._opts is not None else None

    @property
    def bounds(self) -> Optional[Tuple[float, float]]:
        return self._bounds

    @property
    def node(self) -> Optional[Union[DesignVariableNode, ChoiceNode]]:
        return self._node

    def rand(self):
        if self.is_discrete:
            return random.randint(0, self.n_opts-1)
        return random.uniform(*self.bounds)

    def __str__(self):
        if self.is_discrete:
            return 'DV: %s @ %d' % (self.name, self.n_opts)
        return 'DV: %s [%.2f..%.2f]' % (self.name, self.bounds[0], self.bounds[1])

    def __repr__(self):
        return str(self)


class Direction(enum.Enum):
    MIN = -1
    MAX = 1
    LTE = -1
    GTE = 1


class Objective:
    """Class representing an objective."""

    def __init__(self, name: str, direction=Direction.MIN, node: MetricNode = None):
        self._name = name
        self._dir = direction
        self._node = node

    @classmethod
    def from_metric_node(cls, metric_node: MetricNode) -> 'Objective':
        name = metric_node.name
        if metric_node.dir is None:
            raise ValueError(f'Metric node has no direction specified: {name}')

        if metric_node.idx is not None:
            name = f'{name}_{metric_node.idx}'

        direction = Direction.MIN if metric_node.dir <= 0 else Direction.MAX
        return cls(name, direction, node=metric_node)

    @property
    def name(self) -> str:
        return self._name

    @property
    def dir(self) -> Direction:
        return self._dir

    @property
    def sign(self) -> int:
        return self._dir.value

    @property
    def node(self) -> Optional[MetricNode]:
        return self._node

    def __str__(self):
        return 'OBJ: %s (%s)' % (self.name, 'min' if self.sign < 0 else 'max')

    def __repr__(self):
        return str(self)


class Constraint:
    """Class representing an inequality constraint. The direction specifies the side which is considered feasible."""

    def __init__(self, name: str, ref: float = 0., direction=Direction.LTE, node: MetricNode = None):
        self._name = name
        self._ref = ref
        self._dir = direction
        self._node = node

    @classmethod
    def from_metric_node(cls, metric_node: MetricNode) -> 'Constraint':
        name = metric_node.name
        if metric_node.dir is None:
            raise ValueError(f'Metric node has no direction specified: {name}')
        if metric_node.ref is None:
            raise ValueError(f'Metric node has no reference value specified: {name}')

        if metric_node.idx is not None:
            name = f'{name}_{metric_node.idx}'

        direction = Direction.LTE if metric_node.dir <= 0 else Direction.GTE
        return cls(name, metric_node.ref, direction, node=metric_node)

    @property
    def name(self) -> str:
        return self._name

    @property
    def ref(self) -> float:
        return self._ref

    @property
    def dir(self) -> Direction:
        return self._dir

    @property
    def sign(self) -> int:
        return self._dir.value

    @property
    def node(self) -> Optional[MetricNode]:
        return self._node

    def __str__(self):
        return 'CON: %s %s %.2f' % (self.name, '<=' if self.sign < 0 else '>=', self.ref)

    def __repr__(self):
        return str(self)
