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
import numpy as np
from typing import *
from adsg_core.graph.adsg_nodes import *
from adsg_core.optimization.dv_output_defs import *
from cached_property import cached_property
from adsg_core.func_cache import cached_function, clear_func_cache
from adsg_core.graph.adsg import ADSGType
from adsg_core.optimization.hierarchy import *
from adsg_core.optimization.assign_enc.encoding import Encoder
from adsg_core.optimization.assign_enc.selector import EncoderSelector
from adsg_core.optimization.assign_enc.time_limiter import run_timeout
from adsg_core.optimization.assign_enc.assignment_manager import AssignmentManagerBase

__all__ = ['GraphProcessor', 'MetricType', 'SelChoiceEncoderType']


class GraphProcessor:
    """
    Base class for the graph processor: the object that captures all logic for using an architecture graph in an
    optimization problem. A graph processor specifies the optimization problem parameters (design variables, objectives,
    constraints, other output) and contains methods for converting design variables to objectives/constraints/output.

    The graph passed to the processor should be one describing the whole design space, and it should already have
    external functions defined.
    """

    default_encoding_timeout = 10  # sec
    encoders: Dict[SelChoiceEncoderType, Type[HierarchyAnalyzerBase]] = {
        # SelChoiceEncoderType.COMPLETE: SelChoiceEncHierarchyAnalyzer,
        SelChoiceEncoderType.COMPLETE: HierarchyAnalyzer,
        SelChoiceEncoderType.FAST: FastHierarchyAnalyzer,
    }
    _n_combs_cutoff = 1e9

    def __init__(self, graph: ADSGType, encoding_timeout: float = None, encoder_type: SelChoiceEncoderType = None):
        self._graph: ADSGType = self._check_graph(graph)
        self._fixed_values: Dict[int, Union[int, float]] = {}
        self._comb_fixed_mask = None
        self._graph_cache = {}
        self._excluded_cache = set()
        self.encoding_timeout = encoding_timeout or self.default_encoding_timeout
        self._encoder_type = encoder_type
        self._reset_conn_encoder_cache = False

    @property
    def graph(self) -> ADSGType:
        """The ADSG that forms the basis for this optimization problem"""
        return self._graph

    @cached_property
    def _hierarchy_analyzer(self) -> HierarchyAnalyzerBase:
        return self._get_hierarchy_analyzer()

    def _get_hierarchy_analyzer(self) -> HierarchyAnalyzerBase:
        if self._encoder_type is not None:
            return self.encoders[self._encoder_type](self._graph)

        def _do_encode():
            an = self.encoders[SelChoiceEncoderType.COMPLETE](self._graph)

            # Trigger encoding
            assert an.n_combinations >= 0

            return an

        try:
            analyzer = run_timeout(self.encoding_timeout, _do_encode)

        except (TimeoutError, MemoryError):
            analyzer = self.encoders[SelChoiceEncoderType.FAST](self._graph)

        return analyzer

    @property
    def encoder_type(self):
        return self._hierarchy_analyzer.get_encoder_type()

    @cached_property
    def selection_choice_nodes(self):
        return self._hierarchy_analyzer.selection_choice_nodes

    @cached_property
    def _sel_choice_opt_nodes(self):
        return self._hierarchy_analyzer.selection_choice_option_nodes

    @cached_property
    def _sel_choice_is_forced(self):
        return self._hierarchy_analyzer.selection_choice_is_forced

    @cached_property
    def _all_des_var_data(self):
        return self._get_des_vars()

    @property
    def all_des_vars(self) -> List[DesVar]:
        return self._all_des_var_data[0]

    @property
    def _sel_choice_idx_map(self):
        return self._all_des_var_data[1]

    @property
    def _conn_choice_data_map(self):
        return self._all_des_var_data[2]

    @property
    def _existence_infeasibility_mask(self):
        return self._all_des_var_data[3]

    @property
    def _existence_mask(self):
        if self._comb_fixed_mask is not None:
            return self._existence_infeasibility_mask & self._comb_fixed_mask
        return self._existence_infeasibility_mask

    @cached_property
    def all_des_var_idx_map(self):
        return {des_var: i for i, des_var in enumerate(self.all_des_vars)}

    @cached_property
    def all_des_var_nodes(self):
        return {des_var.node for des_var in self.all_des_vars}

    @cached_property
    def all_des_var_node_map(self):
        des_vars = self.all_des_vars
        return {node: [des_var for des_var in des_vars if des_var.node is node] for node in self.all_des_var_nodes}

    @property
    def des_vars(self) -> List[DesVar]:
        """The optimization problem design variables"""
        fixed_values = self._fixed_values
        return [des_var for i, des_var in enumerate(self.all_des_vars) if i not in fixed_values]

    @property
    def dv_is_conditionally_active(self) -> List[bool]:
        return [dv.conditionally_active for dv in self.des_vars]

    @property
    def fixed_values(self):
        return self._fixed_values.copy()

    @cached_property
    def objectives(self) -> List[Objective]:
        """The optimization problem objectives"""
        return self._categorized_metrics[0]

    @cached_property
    def constraints(self) -> List[Constraint]:
        """The optimization problem constraints"""
        return self._categorized_metrics[1]

    @cached_property
    def permanent_nodes(self):
        return self._get_permanent_nodes()

    @cached_property
    def _metrics(self):
        return self._get_metrics()

    @cached_property
    def _categorized_metrics(self):
        return self._categorize_metrics()

    @staticmethod
    def _check_graph(graph: ADSGType) -> ADSGType:
        """
        Check the initially provided graph:
        - Should have external functions defined
        - Should have choices to be made
        """

        # Check if start nodes have been defined
        if graph.derivation_start_nodes is None:
            raise ValueError('The provided graph should have derivation-start nodes defined')

        # # Check if there are any choices
        # if len(graph.get_nodes_by_type(SelectionChoiceNode)) == 0:
        #     if len(graph.choice_nodes) == 0 and len(graph.des_var_nodes) == 0:
        #         raise ValueError('The provided graph contains no choices: there is no optimization to be done!')

        # Check if the initial graph is feasible
        if not graph.feasible:
            raise ValueError('The provided graph is not feasible to begin with!')

        return graph

    def _get_des_vars(self) -> Tuple[List[DesVar], List[int], Dict[ConnectionChoiceNode, tuple], np.ndarray]:
        """
        The design variables for a given architecture graph are defined as follows (and in this order):
        - Each SelectionChoiceNode gets one DesVar
        - Each ConnectionChoiceNode gets one or more DesVars, depending on the encoding scheme
        - Each DesignVariableNode gets one DesVar
        """

        # Check if there are any feasible graphs to start with
        if self._hierarchy_analyzer.n_combinations == 0:
            raise RuntimeError('There are no feasible graphs to begin with!')

        # Create design variables for selection-choice nodes
        des_vars = []
        sel_choice_idx_map = []
        existing_names = set()
        sel_choice_opt_nodes = self._sel_choice_opt_nodes
        is_forced = self._sel_choice_is_forced
        permanent_nodes = self._hierarchy_analyzer.influence_matrix.permanent_nodes_incl_choice_nodes
        for i_dec, choice_node in enumerate(self.selection_choice_nodes):
            if is_forced[i_dec]:
                continue
            options = sel_choice_opt_nodes[choice_node]
            des_vars.append(DesVar.from_choice_node(choice_node, options, existing_names=existing_names,
                                                    conditionally_active=choice_node not in permanent_nodes))
            sel_choice_idx_map.append(i_dec)

            # Track existing names
            existing_names = {des_var.name for des_var in des_vars}

        # Connection choice nodes
        cutoff_mode = False
        n_combs = self._hierarchy_analyzer.n_combinations
        if n_combs > self._n_combs_cutoff:
            n_combs = 1
            cutoff_mode = True

        existence_infeasibility_mask = np.ones((n_combs,), dtype=bool)
        conn_choice_data_map = {}
        for choice_node in self.connection_choice_nodes:
            assignment_manager, conn_des_vars, node_map, exist_map, all_conn_nodes = \
                self._encode_connection_choice(choice_node)
            i_dv_start = len(des_vars)

            for conn_des_var in conn_des_vars:
                if conn_des_var.node not in permanent_nodes:
                    conn_des_var.conditionally_active = True

            des_vars += conn_des_vars
            i_dv_end = len(des_vars)

            # Exclude selection-choice combinations that lead to infeasible port connections
            if not isinstance(exist_map, dict) and not cutoff_mode:
                existence_infeasibility_mask[exist_map == -1] = False

            conn_choice_data_map[choice_node] = \
                (assignment_manager, node_map, exist_map, i_dv_start, i_dv_end, all_conn_nodes)

        # Create design variables for continuous design variable nodes
        des_vars += [DesVar.from_des_var_node(des_var_node, conditionally_active=des_var_node not in permanent_nodes)
                     for des_var_node in self.design_variable_nodes]

        return des_vars, sel_choice_idx_map, conn_choice_data_map, existence_infeasibility_mask

    def fix_des_var(self, des_var: DesVar, value: Union[int, float] = None):
        """
        Fix a design variable to a specific value. Provide `None` to unfix.
        """
        idx = self.all_des_vars.index(des_var)
        if value is None:
            if idx in self._fixed_values:
                del self._fixed_values[idx]
        else:
            if des_var.is_discrete:
                if value < 0 or value >= des_var.n_opts:
                    raise ValueError('Value (%d) out of range: <%d' % (value, des_var.n_opts))
            else:
                if value < des_var.bounds[0] or value > des_var.bounds[1]:
                    raise ValueError('Value out of bounds: %.2f <= %.2f <= %.2f' %
                                     (des_var.bounds[0], value, des_var.bounds[1]))

            # Do not allow for connection choices, because the imputation mechanism allows for the effective selection
            # of other than the fixed value because it is not aware of the fixed nature of the variable
            # If this were to be supported again in the future, uncomment code related to _filter_conn_des_vectors_fixed
            for _, _, _, i_dv_start, i_dv_end, _ in self._conn_choice_data_map.values():
                if i_dv_start <= idx < i_dv_end:
                    raise RuntimeError('Design variable fixing not support for connection choices!')

            self._fixed_values[idx] = value

        self._update_comb_fixed_mask()
        clear_func_cache(self)

    def _update_comb_fixed_mask(self):
        fixed_choices = {}
        for i_dv, i_dec in enumerate(self._sel_choice_idx_map):
            if i_dv not in self._fixed_values:
                continue
            fixed_idx = int(self._fixed_values[i_dv])
            fixed_choices[i_dec] = fixed_idx

        self._comb_fixed_mask = self._hierarchy_analyzer.get_available_combinations_mask(fixed_choices)

    def free_des_var(self, des_var: DesVar):
        """Unfix a design variable"""
        self.fix_des_var(des_var, None)

    def is_fixed(self, des_var: DesVar) -> bool:
        """Returns whether a design variable is fixed"""
        idx = self.all_des_vars.index(des_var)
        return idx in self._fixed_values

    def fixed_value(self, des_var: DesVar) -> Union[float, int]:
        """Get the value of a fixed design variable"""
        idx = self.all_des_vars.index(des_var)
        if idx not in self._fixed_values:
            raise RuntimeError('Design variable not fixed: %r' % des_var)
        return self._fixed_values[idx]

    @staticmethod
    def _get_inactive_value(des_var: DesVar) -> Union[float, int]:
        return X_INACTIVE_IMPUTE if des_var.is_discrete else (sum(des_var.bounds)/2)

    @cached_property
    def connection_choice_nodes(self) -> List[ConnectionChoiceNode]:
        return self.graph.ordered_choice_nodes(
            [node for node in self.graph.choice_nodes if isinstance(node, ConnectionChoiceNode)])

    @cached_property
    def _choice_nodes(self) -> List[ChoiceNode]:
        choice_nodes: List[ChoiceNode] = []
        choice_nodes += self.selection_choice_nodes
        choice_nodes += self.connection_choice_nodes
        return choice_nodes

    @cached_property
    def design_variable_nodes(self) -> List[DesignVariableNode]:
        return self.graph.ordered_choice_nodes(self.graph.des_var_nodes)

    @cached_property
    def metric_nodes(self) -> List[MetricNode]:
        return sorted(self.graph.get_nodes_by_type(MetricNode), key=lambda n: n.name)

    def _get_metrics(self) -> List[Tuple[MetricNode, MetricType]]:
        permanent_nodes = self.permanent_nodes
        metrics = []
        for metric_node in self.metric_nodes:
            is_none = isinstance(metric_node.type, MetricType) and metric_node.type == MetricType.NONE
            if is_none:
                metric_type = MetricType.NONE
            else:
                obj = MetricType.OBJECTIVE if self._can_be_objective(metric_node, permanent_nodes) else MetricType.NONE
                constr = MetricType.CONSTRAINT if self._can_be_constraint(metric_node) else MetricType.NONE

                metric_type = obj | constr

            # If both is possible, check if there is a preference
            if metric_type == MetricType.OBJ_OR_CON and isinstance(metric_node.type, MetricType):
                metric_type = metric_node.type

            metrics.append((metric_node, metric_type))
        return metrics

    def _get_permanent_nodes(self):
        confirmed_initial_graph = self.graph.get_confirmed_graph()
        return set(confirmed_initial_graph.graph.nodes)

    @staticmethod
    def _can_be_objective(metric_node, permanent_nodes):
        """A metric can be an objective if it is part of the permanent nodes (i.e. is always there)."""
        return metric_node.dir is not None and metric_node in permanent_nodes

    @staticmethod
    def _can_be_constraint(metric_node):
        """A metric can be a constraint if a reference value has been defined."""
        return metric_node.dir is not None and metric_node.ref is not None

    def _categorize_metrics(self):
        objectives = []
        constraints = []
        for metric_node, metric_type in self._metrics:
            if metric_type & MetricType.OBJECTIVE:
                objective = Objective.from_metric_node(metric_node)

                # Need to choose?
                if metric_type & MetricType.CONSTRAINT:
                    constraint = Constraint.from_metric_node(metric_node)

                    item = self._choose_metric_type(objective, constraint)
                    if isinstance(item, Objective):
                        objectives.append(item)
                    else:
                        constraints.append(item)

                else:
                    objectives.append(objective)

            elif metric_type & MetricType.CONSTRAINT:
                constraints.append(Constraint.from_metric_node(metric_node))

        return objectives, constraints

    @cached_function
    def get_statistics(self):
        """Computes the following statistics:
        - Number of valid combinatorial design points (1)
        - Number of combinatorial design points (2)
        - Imputation ratio (2) / (1) --> lower is better
        - Information index (0 = 1 variable with n options; 1 = m variables with 2 options) --> higher is better
        - Distance correlation (-1 to 1, 1 means perfect correlation) --> higher is better
        - Number of existence patterns

        for: selection-choices, each connection-choice, additional design variables, total"""
        import pandas as pd

        stats = {'type': [], 'n_valid': [], 'n_declared': [], 'n_discrete': [], 'n_dim_cont': [], 'n_dim_cont_mean': [],
                 'n_exist': [], 'imp_ratio': [], 'imp_ratio_comb': [], 'imp_ratio_cont': [], 'inf_idx': [],
                 'dist_corr': [], 'encoder': []}

        # Selection-choices
        an = self._hierarchy_analyzer
        stats['type'].append('option-decisions')
        stats['n_valid'].append(an.n_combinations)
        stats['n_declared'].append(an.n_design_space)
        stats['n_discrete'].append(len(an.selection_choice_nodes))
        stats['n_dim_cont'].append(0)
        stats['n_dim_cont_mean'].append(0.)
        stats['imp_ratio'].append(an.imputation_ratio)
        stats['imp_ratio_comb'].append(an.imputation_ratio)
        stats['imp_ratio_cont'].append(1.)
        stats['inf_idx'].append(an.information_index)
        stats['dist_corr'].append(0.)
        stats['n_exist'].append(1)
        sel_choice_enc = self.encoder_type.value
        stats['encoder'].append(sel_choice_enc)

        # Permutation-options
        for i, connection_choice in enumerate(self.connection_choice_nodes):
            assignment_manager: AssignmentManagerBase
            assignment_manager, _, _, _, _, _ = self._conn_choice_data_map[connection_choice]
            stats['type'].append(f'permutation-decision-{i} {connection_choice.decision_id}')
            imp_ratio = assignment_manager.encoder.get_imputation_ratio()
            if np.isinf(imp_ratio):
                imp_ratio = 1e6
            stats['imp_ratio'].append(imp_ratio)
            stats['imp_ratio_comb'].append(imp_ratio)
            n_declared = assignment_manager.encoder.get_n_design_points()
            stats['n_declared'].append(n_declared)
            stats['n_discrete'].append(len(assignment_manager.design_vars))
            n_exist = len(list(assignment_manager.matrix_gen.iter_existence()))
            stats['n_exist'].append(n_exist)
            stats['n_valid'].append(int((n_declared*n_exist)/imp_ratio))
            stats['inf_idx'].append(assignment_manager.encoder.get_information_index())
            stats['dist_corr'].append(assignment_manager.encoder.get_distance_correlation())
            stats['encoder'].append(str(assignment_manager.encoder))

            stats['n_dim_cont'].append(0)
            stats['n_dim_cont_mean'].append(0.)
            stats['imp_ratio_cont'].append(1.)

        # Additional design variables
        n_dim_cont, n_dim_cont_mean, imp_ratio_cont = 0, 0., 1.
        if len(self.design_variable_nodes) > 0:
            n_comb_additional_dv, n_declared, n_discrete, n_dim_cont, n_dim_cont_mean, imp_ratio_cont = \
                self.get_additional_dv_stats()
            n_valid = int(np.sum(n_comb_additional_dv))
            if n_valid == 0 or n_discrete == 0:
                imp_ratio_comb = 1.
            else:
                imp_ratio_comb = n_declared*len(n_comb_additional_dv)/n_valid
            imp_ratio = imp_ratio_comb*imp_ratio_cont

            stats['type'].append('additional-dvs')
            stats['n_valid'].append(n_valid)
            stats['n_declared'].append(n_declared)
            stats['n_discrete'].append(n_discrete)
            stats['n_dim_cont'].append(n_dim_cont)
            stats['n_dim_cont_mean'].append(n_dim_cont_mean)
            stats['n_exist'].append(len(n_comb_additional_dv))
            stats['imp_ratio'].append(imp_ratio)
            stats['imp_ratio_comb'].append(imp_ratio_comb)
            stats['imp_ratio_cont'].append(imp_ratio_cont)
            stats['inf_idx'].append(1.)
            stats['dist_corr'].append(0.)
            stats['encoder'].append('')

        # Complete problem
        stats['type'].append('total-design-space')
        stats['n_valid'].append(self.get_n_valid_designs())
        stats['n_declared'].append(self.get_n_design_space())
        stats['n_discrete'].append(len(self.all_des_vars)-n_dim_cont)
        stats['n_dim_cont'].append(n_dim_cont)
        stats['n_dim_cont_mean'].append(n_dim_cont_mean)
        stats['imp_ratio'].append(self.get_imputation_ratio())
        stats['imp_ratio_comb'].append(self.get_imputation_ratio(include_cont=False))
        stats['imp_ratio_cont'].append(imp_ratio_cont)
        stats['inf_idx'].append(self.get_information_index())
        stats['dist_corr'].append(0.)
        stats['n_exist'].append(1)
        stats['encoder'].append(sel_choice_enc)

        # Problem with fixed design variables
        _, _, _, n_dim_cont_fix, n_dim_cont_mean_fix, imp_ratio_cont_fix = self.get_additional_dv_stats(with_fixed=True)
        stats['type'].append('total-design-problem')
        stats['n_valid'].append(self.get_n_valid_designs(with_fixed=True))
        stats['n_declared'].append(self.get_n_design_space(with_fixed=True))
        stats['n_discrete'].append(len(self.des_vars)-n_dim_cont_fix)
        stats['n_dim_cont'].append(n_dim_cont_fix)
        stats['n_dim_cont_mean'].append(n_dim_cont_mean_fix)
        stats['imp_ratio'].append(self.get_imputation_ratio(with_fixed=True))
        stats['imp_ratio_comb'].append(self.get_imputation_ratio(include_cont=False, with_fixed=True))
        stats['imp_ratio_cont'].append(imp_ratio_cont_fix)
        stats['inf_idx'].append(self.get_information_index(with_fixed=True))
        stats['dist_corr'].append(0.)
        stats['n_exist'].append(1)
        stats['encoder'].append(sel_choice_enc)

        df = pd.DataFrame(data=stats)
        df = df.set_index('type')
        return df

    def print_stats(self):
        """Get and print statistics"""
        import pandas as pd
        with pd.option_context('display.max_rows', None, 'display.max_columns', None,
                               'display.expand_frame_repr', False, 'max_colwidth', None):
            print(self.get_statistics())

    def _get_dv_n_opts(self, with_fixed=False, include_cont=False):
        des_vars = self.des_vars if with_fixed else self.all_des_vars
        if not include_cont:
            return [dv.n_opts for dv in des_vars if dv.is_discrete]
        return [dv.n_opts if dv.is_discrete else 2 for dv in des_vars]

    def get_n_design_space(self, with_fixed=False, include_cont=False) -> int:
        """Get the declared design space size (Cartesian product of discrete variables)"""
        n_opts = self._get_dv_n_opts(with_fixed=with_fixed, include_cont=include_cont)
        if len(n_opts) == 0:
            return 1
        return int(np.prod(n_opts, dtype=float))

    def get_information_index(self, with_fixed=False) -> float:
        return Encoder.calc_information_index(self._get_dv_n_opts(with_fixed=with_fixed))

    @cached_function
    def get_n_valid_designs(self, with_fixed=False, include_cont=False) -> int:
        """Get the number of valid (discrete) architectures"""
        # Possible combinations of selection-choices
        cutoff_mode = False
        n_combs = self._hierarchy_analyzer.n_combinations
        if n_combs > self._n_combs_cutoff:
            cutoff_mode = True
            n_combinations = np.ones((1,), dtype=float)*n_combs
        else:
            n_combinations = np.ones((n_combs,), dtype=float)
        if len(n_combinations) == 0:
            return 0

        # Possible combinations of independent connection-choices, per selection-choice combination
        for connection_choice in self.connection_choice_nodes:
            assignment_manager: AssignmentManagerBase
            assignment_manager, _, exist_map, i_dv_start, i_dv_end, _ = self._conn_choice_data_map[connection_choice]

            # If no accurate node existence information is available, assume the most combinations possible
            if isinstance(exist_map, dict) or cutoff_mode:
                n_comb_max = assignment_manager.matrix_gen.count_all_matrices(max_by_existence=True)
                n_combinations *= n_comb_max

            else:
                matrix_map = assignment_manager.matrix_gen.get_agg_matrix(cache=True)
                # dv_map = assignment_manager.get_all_design_vectors() if with_fixed else {}
                for i_comb, i_exist in enumerate(exist_map):
                    if i_exist == -1:  # Infeasible existence scheme
                        n_combinations[i_comb] *= 0
                    else:
                        existence = assignment_manager.matrix_gen.existence_patterns.patterns[i_exist]
                        # if with_fixed:
                        #     des_vectors = self._filter_conn_des_vectors_fixed(
                        #         dv_map[existence], i_dv_start, i_dv_end, self._fixed_values)
                        #     n_dvs = des_vectors.shape[0]
                        # else:
                        n_dvs = matrix_map[existence].shape[0]
                        n_combinations[i_comb] *= n_dvs

        # Possible combinations of additional design variable values, per selection-choice combination
        if len(self.design_variable_nodes) > 0:
            n_comb_additional_dv, _, _, _, _, _ = self.get_additional_dv_stats(
                with_fixed=with_fixed, cont_as_discrete=include_cont)
            n_combinations *= n_comb_additional_dv

        # Determine which selection-choice combinations we should count
        if with_fixed and self._comb_fixed_mask is not None:
            n_combinations = n_combinations[self._comb_fixed_mask]

        return int(np.sum(n_combinations))

    # @staticmethod
    # def _filter_conn_des_vectors_fixed(des_vectors: np.ndarray, i_dv_start, i_dv_end, fixed_values: dict):
    #     for i, i_dv in enumerate(range(i_dv_start, i_dv_end)):
    #         fixed_value = fixed_values.get(i_dv)
    #         if fixed_value is not None:
    #             matched_mask = des_vectors[:, i] == fixed_value
    #
    #             # If the fixed value is zero, also match the inactive value (-1), because that is represented as zero
    #             if fixed_value == 0:
    #                 matched_mask |= des_vectors[:, i] == -1
    #
    #             des_vectors = des_vectors[matched_mask, :]
    #     return des_vectors

    @cached_function
    def get_all_discrete_x(self, with_fixed=True) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Returns all possible discrete design vectors. Warning: contains no limits on time or size!"""
        fixed_values = self._fixed_values if with_fixed else {}
        dv_idx_map = self.all_des_var_idx_map
        node_map = self.all_des_var_node_map
        n_dv = len(self.all_des_vars)

        # Selection choice design vectors
        choice_opt_idx = self._hierarchy_analyzer.get_choice_option_indices()
        if choice_opt_idx is None:
            return
        dv_sel = choice_opt_idx[:, ~self._sel_choice_is_forced]
        x = -np.ones((dv_sel.shape[0], n_dv), dtype=int)
        x[:, :dv_sel.shape[1]] = dv_sel[:, :]
        i_combs = np.arange(x.shape[0])

        # Filter fixed and infeasible values
        x_keep = self._existence_mask if with_fixed else self._existence_infeasibility_mask
        x = x[x_keep, :]
        i_combs = i_combs[x_keep]

        def _expand_at(i_dv_expand, x_expand, i_comb_expand):
            nonlocal x, i_combs
            n_repeat = np.ones((x.shape[0],), dtype=int)
            is_expanded = np.zeros((x.shape[0],), dtype=bool)
            n_expand = x_expand.shape[0]
            for i_comb_active in i_comb_expand:
                i_comb_act_mask = i_combs == i_comb_active
                n_repeat[i_comb_act_mask] = n_expand
                is_expanded[i_comb_act_mask] = True

            if ~np.any(is_expanded):
                return
            x = np.repeat(x, n_repeat, axis=0)
            i_combs = np.repeat(i_combs, n_repeat)

            x_columns = x[:, i_dv_expand]
            cumulative_idx = np.cumsum(n_repeat)-n_repeat
            for i_set, cum_idx in enumerate(cumulative_idx):
                if is_expanded[i_set]:
                    x_columns[cum_idx:cum_idx+n_expand, :] = x_expand

            x[:, i_dv_expand] = x_columns

        # Connection choices
        conn_choice_data_map = self._conn_choice_data_map
        conn_choice_existence = self._hierarchy_analyzer.get_nodes_existence(self.connection_choice_nodes)
        if conn_choice_existence is None:
            return
        for i_node, choice_node in enumerate(self.connection_choice_nodes):
            i_is_active = conn_choice_existence[:, i_node]

            assignment_manager: AssignmentManagerBase
            assignment_manager, conn_node_map, exist_map, i_dv_start, i_dv_end, all_conn_nodes = \
                conn_choice_data_map[choice_node]

            x_conn_map = assignment_manager.get_all_design_vectors()
            existence_patterns = assignment_manager.matrix_gen.existence_patterns.patterns

            if isinstance(exist_map, dict):
                raise RuntimeError('Unexpected existence map type')
            for i_exist in np.unique(exist_map):
                if i_exist == -1:
                    continue
                i_comb_exist = np.where((exist_map == i_exist) & i_is_active)[0]
                if len(i_comb_exist) == 0:
                    continue

                existence = existence_patterns[i_exist]
                x_existence = x_conn_map[existence]
                # x_existence = self._filter_conn_des_vectors_fixed(x_existence, i_dv_start, i_dv_end, fixed_values)

                _expand_at(np.arange(i_dv_start, i_dv_end), x_existence, i_comb_exist)

        # Additional discrete design variables
        dv_node_existence = self._hierarchy_analyzer.get_nodes_existence(self.design_variable_nodes)
        if dv_node_existence is None:
            return
        for i_dv, des_var_node in enumerate(self.design_variable_nodes):
            i_is_active = np.where(dv_node_existence[:, i_dv])[0]
            des_var = node_map[des_var_node][0]
            dv_idx = dv_idx_map[des_var]

            if not des_var.is_discrete:
                for i_comb in i_is_active:
                    x[i_combs == i_comb, dv_idx] = 0
                continue

            values = np.array([np.arange(des_var.n_opts)]).T
            if dv_idx in fixed_values:
                values = values[[fixed_values[dv_idx]], :]
            _expand_at([dv_idx], values, i_is_active)

        # Get activeness
        is_active = x != X_INACTIVE_VALUE  # Applies both for the hierarchy analyzer and assignment encoders
        x = x.astype(float)
        for i_dv, dv in enumerate(self.all_des_vars):
            inactive_value = self._get_inactive_value(dv)
            x[x[:, i_dv] == X_INACTIVE_VALUE, i_dv] = inactive_value

        # Return only non-fixed design variables
        is_free = np.array([i for i in range(n_dv) if i not in fixed_values], dtype=int)
        x = x[:, is_free]
        is_active = is_active[:, is_free]
        return x, is_active

    @cached_function
    def get_additional_dv_stats(self, with_fixed=False, cont_as_discrete=False):
        cutoff_mode = False
        n_comb = self._hierarchy_analyzer.n_combinations
        multiplier = 1
        if n_comb > self._n_combs_cutoff:
            multiplier = n_comb
            n_comb = 1
            cutoff_mode = True

        if len(self.design_variable_nodes) == 0:
            nodes_existence = np.zeros((n_comb, 0), dtype=bool)
        else:
            try:
                nodes_existence = self._hierarchy_analyzer.get_nodes_existence(self.design_variable_nodes) \
                    if not cutoff_mode else None
            except MemoryError:
                nodes_existence = None
                multiplier = n_comb
                n_comb = 1

            # If precise information is unavailable, assume all nodes are active
            if nodes_existence is None:
                nodes_existence = np.ones((n_comb, len(self.design_variable_nodes)), dtype=bool)*multiplier

        cont_mask = np.array([not node.is_discrete for node in self.design_variable_nodes], dtype=bool)
        n_opts = []
        n_comb_additional_dv = np.ones(nodes_existence.shape, dtype=int)
        for i_dv_node, dv_node in enumerate(self.design_variable_nodes):
            # Do not count if fixed
            i_dv = self.all_des_vars.index(self.all_des_var_node_map[dv_node][0])
            if with_fixed and i_dv in self._fixed_values:
                n_opts.append(1)
                continue

            if not cont_mask[i_dv_node] or cont_as_discrete:
                n_combs_dv = 2 if cont_mask[i_dv_node] else len(dv_node.options)
                n_comb_additional_dv[nodes_existence[:, i_dv_node], i_dv_node] = n_combs_dv
                n_opts.append(n_combs_dv)
            else:
                n_comb_additional_dv[~nodes_existence[:, i_dv_node], i_dv_node] = 0

        n_comb_discrete = n_comb_additional_dv if cont_as_discrete else n_comb_additional_dv[:, ~cont_mask]
        n_discrete = n_comb_discrete.shape[1]
        if n_comb_discrete.shape[1] == 0:
            n_comb_additional_dv_cum = np.ones((nodes_existence.shape[0],), dtype=int)
            n_declared = 0
        else:
            n_comb_additional_dv_cum = np.prod(n_comb_discrete, axis=1, dtype=float)
            n_declared = int(np.prod(n_opts, dtype=float)) if len(n_opts) > 0 else 1

        n_comb_cont = n_comb_additional_dv[:, cont_mask]
        if n_comb_cont.shape[1] == 0:
            n_dim_cont, n_dim_cont_mean, imp_ratio_cont = 0, 0., 1.
        else:
            n_dim_cont = n_comb_cont.shape[1]
            n_dim_cont_mean = np.sum(n_comb_cont)/n_comb_cont.shape[0]
            imp_ratio_cont = n_dim_cont/n_dim_cont_mean if n_dim_cont_mean > 0 else 1.

        return n_comb_additional_dv_cum, n_declared, n_discrete, n_dim_cont, n_dim_cont_mean, imp_ratio_cont

    def get_imputation_ratio(self, with_fixed=False, include_cont=True) -> float:
        """
        Get the imputation ratio: the ratio between the declared and valid design space sizes.
        A value of 1 indicates there is no design space hierarchy, values greater than 1 indicate hierarchy.
        """
        n_valid = self.get_n_valid_designs(with_fixed=with_fixed)
        if n_valid == 0:
            return 1.
        imp_ratio = self.get_n_design_space(with_fixed=with_fixed)/n_valid

        if include_cont:
            _, _, _, _, _, imp_ratio_cont = self.get_additional_dv_stats(with_fixed=with_fixed)
            imp_ratio *= imp_ratio_cont

        return imp_ratio

    def get_random_design_vector(self) -> Sequence[Union[int, float]]:
        """Generate a random design vector"""
        return [dv.rand() for dv in self.des_vars]

    def _get_all_des_var_values(self, des_var_values: Sequence[Union[int, float]]) -> List[Union[int, float]]:
        fixed_values = self._fixed_values
        i_value = 0
        values = []
        for i, des_var in enumerate(self.all_des_vars):
            if i in fixed_values:
                values.append(fixed_values[i])
            else:
                values.append(des_var_values[i_value])
                i_value += 1
        return values

    def get_graph(self, des_var_values: Sequence[Union[int, float]], create=True) \
            -> Tuple[ADSGType, List[Union[int, float]], List[bool]]:
        """Creates an ADSG instance from a design vector; optionally only returning the imputed design vector and
        activeness vector without actually creating the graph"""

        if len(self.des_vars) != len(des_var_values):
            raise ValueError('Incorrect number of design variable values: %d != %d' %
                             (len(des_var_values), len(self.des_vars)))

        des_var_values_in = des_var_values
        des_var_values = self._get_all_des_var_values(des_var_values)
        des_vars = self.all_des_vars
        if len(des_vars) != len(des_var_values):
            raise RuntimeError

        # Get graph for the selection-choice nodes
        sel_choice_opt_idx = [0 for _ in range(len(self.selection_choice_nodes))]
        is_fixed = [False for _ in range(len(sel_choice_opt_idx))]
        for i_dv, i_dec in enumerate(self._sel_choice_idx_map):
            sel_choice_opt_idx[i_dec] = int(des_var_values[i_dv])
            is_fixed[i_dec] = i_dv in self._fixed_values

        sel_choice_is_active = graph_instance = i_comb = None
        try:
            if not create:
                graph_instance = None
                sel_choice_opt_idx, sel_choice_is_active, i_comb = self._hierarchy_analyzer.get_opt_idx(
                    sel_choice_opt_idx, mask=self._existence_mask, is_fixed=is_fixed, exclude=self._excluded_cache)

                # If the combination idx is not known (fast encoder),
                # we need to actually create the graph to know which nodes exist
                if i_comb is None:
                    create = True

            if create:
                graph_instance, sel_choice_opt_idx, sel_choice_is_active, i_comb = self._hierarchy_analyzer.get_graph(
                    sel_choice_opt_idx, mask=self._existence_mask, is_fixed=is_fixed, exclude=self._excluded_cache)

        except RuntimeError:
            print(f'Error occurred while getting graph for DV: {des_var_values}')
            raise

        opt_dec_used_values: List[Optional[int]] = \
            [int(val) for val in list(np.array(sel_choice_opt_idx)[self._sel_choice_idx_map])]
        for i_dv, i_dec in enumerate(self._sel_choice_idx_map):
            if not sel_choice_is_active[i_dec]:
                opt_dec_used_values[i_dv] = None

        opt_dec_existence_key = tuple(sel_choice_opt_idx)

        # Associate rest of design variable to choice nodes
        idx_map = self.all_des_var_idx_map
        node_map = self.all_des_var_node_map

        # Track actually used design variable values
        used_values: List[Optional[Union[int, float]]] = [None for _ in range(len(des_vars))]
        used_values[:len(opt_dec_used_values)] = opt_dec_used_values

        # Make connection-choices
        graph_cache = self._graph_cache
        prev_values = opt_dec_used_values
        conn_choice_data_map = self._conn_choice_data_map

        if i_comb is not None:
            conn_choice_existence = \
                self._hierarchy_analyzer.get_nodes_existence(self.connection_choice_nodes, i_comb=i_comb)
        elif graph_instance is not None:
            conn_choice_existence = [choice_node in graph_instance.choice_nodes
                                     for choice_node in self.connection_choice_nodes]
        else:
            raise RuntimeError('Either combination idx or graph instance is needed!')

        for i_conn_choice, choice_node in enumerate(self.connection_choice_nodes):
            # Check if choice still exists
            if not conn_choice_existence[i_conn_choice]:
                continue
            if graph_instance is not None and choice_node not in graph_instance.choice_nodes:
                raise RuntimeError('Connection choice not does not exist!')

            assignment_manager: AssignmentManagerBase
            assignment_manager, conn_node_map, exist_map, i_dv_start, i_dv_end, all_conn_nodes = \
                conn_choice_data_map[choice_node]

            if isinstance(exist_map, dict):
                if graph_instance is None:
                    raise RuntimeError('Graph instance expected!')
                existing_nodes = graph_instance.graph.nodes
                nodes_exist_mask = tuple(node in existing_nodes for node in all_conn_nodes)
                i_exist_pattern = exist_map[nodes_exist_mask]
            else:
                i_exist_pattern = exist_map[i_comb]

            # If the pattern has no solutions, either try to find another graph or raise error if this shouldn't happen
            if i_exist_pattern == -1:
                if i_comb is None:
                    self._excluded_cache.add(tuple(sel_choice_opt_idx))
                    return self.get_graph(des_var_values_in, create=create)
                else:
                    raise RuntimeError('Infeasible graph specified!')

            choice_des_vector = np.array(des_var_values[i_dv_start:i_dv_end]).astype(int)
            existence = assignment_manager.matrix_gen.existence_patterns.patterns[i_exist_pattern]
            choice_des_vector, choice_is_active, conn_edges = \
                assignment_manager.get_conn_idx(choice_des_vector, existence=existence)

            node_edges = [(conn_node_map[0][conn_edge[0]], conn_node_map[1][conn_edge[1]]) for conn_edge in conn_edges]

            if graph_instance is not None:
                cache_key = (choice_node, opt_dec_existence_key, tuple(choice_des_vector), tuple(prev_values))
                if cache_key in graph_cache:
                    graph_instance = graph_cache[cache_key].copy()
                else:
                    graph_cache[cache_key] = graph_instance = \
                        graph_instance.get_for_apply_connection_choice(choice_node, node_edges, validate=False)

            used_values[i_dv_start:i_dv_end] = [
                int(val) if choice_is_active[i_dv] else None for i_dv, val in enumerate(choice_des_vector)]
            prev_values = prev_values+list(choice_des_vector)

        # Set values of additional design variables
        if i_comb is not None:
            dv_node_existence = self._hierarchy_analyzer.get_nodes_existence(self.design_variable_nodes, i_comb=i_comb)
        elif graph_instance is not None:
            dv_node_existence = [dv_node in graph_instance.des_var_nodes for dv_node in self.design_variable_nodes]
        else:
            raise RuntimeError('Either combination idx or graph instance is needed!')

        if np.any(dv_node_existence):
            if graph_instance is not None:
                graph_instance = graph_instance.copy()
            for i_dv, des_var_node in enumerate(self.design_variable_nodes):
                if not dv_node_existence[i_dv]:
                    continue
                if graph_instance is not None and des_var_node not in graph_instance.des_var_nodes:
                    raise RuntimeError('Des var node not found!')
                if des_var_node not in node_map:
                    raise ValueError('Unexpected design variable node: %s' % des_var_node.str_context())
                des_var = node_map[des_var_node][0]
                dv_idx = idx_map[des_var]
                des_var_value = des_var_values[dv_idx]

                des_var_value = int(des_var_value) if des_var.is_discrete else float(des_var_value)
                if graph_instance is not None:
                    graph_instance.set_des_var_value(des_var_node, des_var_value)
                    used_values[dv_idx] = graph_instance.des_var_value(des_var_node)
                else:
                    used_values[dv_idx], _ = des_var_node.correct_value(des_var_value)

        # Set all unused design variables to their inactive values
        is_active = [used_value is not None for used_value in used_values]
        for i, used_value in enumerate(used_values):
            if used_value is None:
                used_values[i] = self._get_inactive_value(des_vars[i])

        # Get non-fixed used values
        used_values = [value for i, value in enumerate(used_values) if i not in self._fixed_values]
        is_active = [is_act for i, is_act in enumerate(is_active) if i not in self._fixed_values]

        return graph_instance, used_values, is_active

    def _encode_connection_choice(self, connection_choice_node: ConnectionChoiceNode):

        # Get all nodes involved in the connection and their existence specifications
        settings, node_map, existence_map, all_conn_nodes = connection_choice_node.get_assignment_encoding_args(
            self.graph, hierarchy_analyzer=self._hierarchy_analyzer)

        # Get assignment manager by automatically selecting the best encoder
        selector = EncoderSelector(settings)
        if self._reset_conn_encoder_cache:
            selector.reset_cache()
        assignment_manager = selector.get_best_assignment_manager()

        # Remove any existence pattern that results in an infeasible graph (i.e. no connections are possible)
        agg_matrix = assignment_manager.matrix_gen.get_agg_matrix(cache=True)
        for i_exist, existence in enumerate(assignment_manager.matrix_gen.existence_patterns.patterns):
            if agg_matrix[existence].shape[0] == 0:
                if isinstance(existence_map, dict):
                    existence_map = {exist_mask: -1 if i_pattern == i_exist else i_pattern
                                     for exist_mask, i_pattern in existence_map.items()}
                else:
                    existence_map[existence_map == i_exist] = -1

        # Convert to design variables
        base_name = connection_choice_node.decision_id
        des_vars = []
        for i, dv in enumerate(assignment_manager.design_vars):
            options = list(range(dv.n_opts))
            name = f'{base_name}_{i}'
            des_vars.append(DesVar.from_choice_node(
                connection_choice_node, options=options, name=name, conditionally_active=dv.conditionally_active))

        return assignment_manager, des_vars, node_map, existence_map, all_conn_nodes

    def _choose_metric_type(self, objective: Objective, constraint: Constraint) -> Union[Objective, Constraint]:
        raise RuntimeError('Cannot choose whether to interpret metric as objective or as constraint: '
                           '%r' % objective.name)
