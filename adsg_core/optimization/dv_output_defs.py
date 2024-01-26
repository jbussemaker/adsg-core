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
        """Design variable name"""
        return self._name

    @property
    def is_discrete(self) -> bool:
        """Whether the design variable is discrete or continuous"""
        return self._opts is not None

    @property
    def options(self) -> Optional[list]:
        """List of options (if discrete)"""
        return self._opts

    @property
    def n_opts(self) -> Optional[int]:
        """Number of options (if discrete)"""
        return len(self._opts) if self._opts is not None else None

    @property
    def bounds(self) -> Optional[Tuple[float, float]]:
        """Bounds (if continuous)"""
        return self._bounds

    @property
    def node(self) -> Optional[Union[DesignVariableNode, ChoiceNode]]:
        """Associated choice node"""
        return self._node

    def rand(self):
        """Generate a random value"""
        if self.is_discrete:
            return random.randint(0, self.n_opts-1)
        return random.uniform(*self.bounds)

    def __str__(self):
        if self.is_discrete:
            return f'DV: {self.name} [{self.n_opts} opts]'
        return f'DV: {self.name} [{self.bounds[0]:.2f}..{self.bounds[1]:.2f}]'

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
        """Objective name"""
        return self._name

    @property
    def dir(self) -> Direction:
        return self._dir

    @property
    def sign(self) -> int:
        """Objective direction (-1 = minimization, 1 = maximization)"""
        return self._dir.value

    @property
    def node(self) -> Optional[MetricNode]:
        """Associated metric node"""
        return self._node

    def __str__(self):
        return f'OBJ: {self.name} [{"min" if self.sign < 0 else "max"}]'

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
        """Objective name"""
        return self._name

    @property
    def ref(self) -> float:
        """Reference value (the value should be above/below this value)"""
        return self._ref

    @property
    def dir(self) -> Direction:
        return self._dir

    @property
    def sign(self) -> int:
        """Constraint direction (-1 = lower than or equal, 1 = greater than or equal)"""
        return self._dir.value

    @property
    def node(self) -> Optional[MetricNode]:
        """Associated metric node"""
        return self._node

    def __str__(self):
        return f'CON: {self.name} [{"<=" if self.sign < 0 else ">="} {self.ref:.2f}]'

    def __repr__(self):
        return str(self)
