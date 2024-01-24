from typing import *
from adsg_core.graph.adsg import *
from adsg_core.graph.sup.nodes import *
from adsg_core.graph.adsg_basic import *
from adsg_core.graph.adsg_nodes import *
from adsg_core.graph.graph_edges import *

__all__ = [
    'SupDSG', 'EdgeType', 'ADSGType', 'SupNode',
    'ConnNodes', 'SelectionChoiceNode', 'ConnectionChoiceNode', 'ConnectorNode', 'ConnectorDegreeGroupingNode',
    'SupChoiceMapping', 'SupInitializationError', 'SupResolveError', 'SupSelChoiceOptionMapping',
    'SupExistenceMapping',
]


class SupDSG(BasicADSG):
    """
    Supplementary Design Space Graph: a DSG where all choices are fully derived from another ADSG (the source DSG).
    Can be useful for modeling variability in graph structures other than the source ADSG.

    Usage:
    1. Add edges and choices (add_edge, add_edges, add_selection_choice, add_connection_choice)
    2. Add choice mappings (add_mapping)
    3. Set start nodes and initialize choices (set_start_nodes)
    """

    def __init__(self, *args, choice_mappings=None, **kwargs):
        self._choice_mappings: List[Tuple[ChoiceNode, SupChoiceMapping]] = choice_mappings or []
        super().__init__(*args, **kwargs)

    def _mod_graph_adjust_kwargs(self, kwargs):
        super()._mod_graph_adjust_kwargs(kwargs)
        kwargs['choice_mappings'] = self._choice_mappings

    def _mod_graph_inplace(self, kwargs):
        if 'choice_mappings' in kwargs:
            self._choice_mappings = kwargs['choice_mappings']

    @property
    def sup_nodes(self) -> List[SupNode]:
        return self.get_nodes_by_type(SupNode)

    def get_by_ref(self, ref) -> Optional[SupNode]:
        for node in self.sup_nodes:
            if node.ref == ref:
                return node

    @property
    def choice_mappings(self) -> List[Tuple[ChoiceNode, 'SupChoiceMapping']]:
        return self._choice_mappings

    def add_mapping(self, sup_choice_node: ChoiceNode, src_dsg: ADSG, choice_mapping: 'SupChoiceMapping'):
        """Add a choice mapping to the Supplementary DSG for a given source DSG"""
        if sup_choice_node not in self.graph.nodes:
            raise RuntimeError(f'Choice node not found: {sup_choice_node!r}')

        choice_mapping.initialize(self, sup_choice_node, src_dsg)
        self._choice_mappings.append((sup_choice_node, choice_mapping))

    def initialize_choices(self):
        mapped_choice_nodes = set()
        dup_mapped = []
        for choice_node, _ in self._choice_mappings:
            if choice_node in mapped_choice_nodes:
                dup_mapped.append(choice_node)
            else:
                mapped_choice_nodes.add(choice_node)

        if len(dup_mapped) > 0:
            raise RuntimeError(f'Duplicate mapped choice nodes: {dup_mapped!r}')

        unmapped_choice_nodes = set(self.choice_nodes) - mapped_choice_nodes
        if len(unmapped_choice_nodes):
            raise RuntimeError(f'Unmapped choice nodes: {unmapped_choice_nodes!r}')

        return super().initialize_choices()

    def resolve(self, src_dsg: ADSG) -> 'SupDSG':
        """Resolve choice mappings from a finalized source DSG"""
        if not src_dsg.final or not src_dsg.feasible:
            raise RuntimeError('Expecting a final and feasible source DSG')

        sup_dsg = self
        for choice_node, choice_mapping in self.choice_mappings:
            # Resolve if choice not is active in the SupDSG
            if choice_node in sup_dsg.graph.nodes:
                sup_dsg = choice_mapping.resolve(sup_dsg, choice_node, src_dsg)

        if not sup_dsg.final:
            raise RuntimeError('Resolved SupDSG is not final; choice nodes remain!')
        return sup_dsg

    def constrain_choices(self, *args, **kwargs):
        raise RuntimeError('Supplementary DSG does not support choice constraints')

    def add_incompatibility_constraint(self, *args, **kwargs):
        raise RuntimeError('Supplementary DSG does not support incompatibility constraints')


class SupInitializationError(RuntimeError):

    def __init__(self, mapping: 'SupChoiceMapping', sup_dsg: SupDSG, source_dsg: ADSG, msg: str):
        self.mapping = mapping
        self.sup_dsg = sup_dsg
        self.source_dsg = source_dsg
        super().__init__(msg)


class SupResolveError(RuntimeError):

    def __init__(self, mapping: 'SupChoiceMapping', sup_dsg: SupDSG, source_dsg: ADSG, msg: str):
        self.mapping = mapping
        self.sup_dsg = sup_dsg
        self.source_dsg = source_dsg
        super().__init__(msg)


class SupChoiceMapping:
    """
    Base class for defining a choice mapping in a supplementary design space graph.
    """

    def initialize(self, sup_dsg: SupDSG, sup_choice_node: ChoiceNode, src_dsg: ADSG):
        """
        Called after adding the choice mapping to a Supplementary DSG.
        Supplementary DSG and source DSG are in their base states (including choices etc.).
        Should raise a SupInitializationError if it cannot be initialized.
        """

    def resolve(self, sup_dsg: SupDSG, sup_choice_node: ChoiceNode, src_dsg: ADSG) -> SupDSG:
        """
        Given the associated choice node in a Supplementary DSG, should resolve the mapping from the source DSG.
        Supplementary DSG and source DSG may be in partially-resolve states already.
        Returns the resolved Supplementary DSG.
        Should raise a SupResolveError if it cannot be resolved.
        """
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class SupSelChoiceOptionMapping(SupChoiceMapping):
    """
    Maps each option of a selection choice of the source DSG to an option of a selection choice of a supplementary DSG.

    `None` as a mapping source value indicates the situation where the choice in the source DSG was not active (i.e. the
    originating node does not exist anymore in the source DSG).
    """

    def __init__(self, src_choice_node: SelectionChoiceNode, mapping: Dict[Optional[ADSGNode], ADSGNode]):
        self._src_choice_originating_node = None
        self._src_choice_node = src_choice_node
        self._mapping = mapping

    def initialize(self, sup_dsg: SupDSG, sup_choice_node: ChoiceNode, src_dsg: ADSG):
        if not isinstance(sup_choice_node, SelectionChoiceNode):
            raise SupInitializationError(self, sup_dsg, src_dsg, f'Expecting SelectionChoiceNode: {sup_choice_node!r}')

        # Check if all source choice option nodes are mapped
        src_choice_node = self._src_choice_node
        if src_choice_node not in src_dsg.graph.nodes:
            raise SupInitializationError(
                self, sup_dsg, src_dsg, f'Source choice node not in source DSG: {src_choice_node!r}')
        src_option_nodes = src_dsg.get_option_nodes(src_choice_node)

        mapping_nodes = {key for key in self._mapping.keys() if key is not None}
        unmapped_src_opt_nodes = set(src_option_nodes) - mapping_nodes
        if len(unmapped_src_opt_nodes) > 0:
            raise SupInitializationError(
                self, sup_dsg, src_dsg, f'Not all source choice option nodes mapped: {unmapped_src_opt_nodes!r}')

        if src_dsg.has_conditional_existence(src_choice_node) and None not in self._mapping:
            raise SupInitializationError(self, sup_dsg, src_dsg,
                                         f'Choice may be inactive; `None` missing from mapping!')

        # Check if all target sup option nodes are available
        sup_option_nodes = sup_dsg.get_option_nodes(sup_choice_node)
        unknown_sup_opt_nodes = set(self._mapping.values()) - set(sup_option_nodes)
        if len(unknown_sup_opt_nodes) > 0:
            raise SupInitializationError(
                self, sup_dsg, src_dsg, f'Not all target nodes are choice option nodes: {unknown_sup_opt_nodes!r}')

        # Get originating node in source DSG
        originating_nodes = [edge[0] for edge in iter_in_edges(src_dsg.graph, src_choice_node)]
        self._src_choice_originating_node = originating_nodes[0]

    def resolve(self, sup_dsg: SupDSG, sup_choice_node: ChoiceNode, src_dsg: ADSG) -> SupDSG:
        assert isinstance(sup_choice_node, SelectionChoiceNode)
        src_originating_node = self._src_choice_originating_node
        assert src_originating_node is not None

        # Determine which option has been selected
        src_nodes = {node.str_context() for node in src_dsg.graph.nodes if isinstance(node, ADSGNode)}
        mapping = self._mapping
        if src_originating_node.str_context() not in src_nodes:
            if None not in mapping:
                raise SupResolveError(
                    self, sup_dsg, src_dsg, f'Could not resolve {mapping!r}: {self._src_choice_node!r} is inactive, '
                                            f'but `None` is missing from the mapping')
            sup_tgt_option_node = mapping[None]

        else:
            originating_out_nodes = {
                edge[1] for edge in iter_out_edges(src_dsg.graph, src_originating_node, edge_type=EdgeType.DERIVES)}
            src_selected_opt_nodes = originating_out_nodes & set(mapping.keys())
            if len(src_selected_opt_nodes) != 1:
                raise SupResolveError(
                    self, sup_dsg, src_dsg, f'Could not resolve {mapping!r}: could not determine which option node '
                                            f'was selected ({src_selected_opt_nodes!r})')

            mapping_ctx = {node.str_context(): sup_node for node, sup_node in mapping.items()}
            sup_tgt_option_node = mapping_ctx[list(src_selected_opt_nodes)[0].str_context()]

        # Apply selection choice node
        return sup_dsg.get_for_apply_selection_choice(sup_choice_node, sup_tgt_option_node)

    def __str__(self):
        mapping_str = '; '.join([f'{src_node!s} -> {tgt_node!s}' for src_node, tgt_node in self._mapping.items()])
        return f'Sel Choice Opt Mapping {self._src_choice_node!s}: {mapping_str}'

    def __repr__(self):
        return f'{self.__class__.__name__}(src_choice_node={self._src_choice_node!r}, mapping={self._mapping!r})'


class SupExistenceMapping(SupChoiceMapping):
    """
    Maps any node in the source DSG to options of a selection choice of a supplementary DSG. Determines which option to
    select based on source DSG node existence. Since node existence is not mutually-exclusive, the order in which
    mappings are declared determines selection priority:
    - If the first source node is selected, the first option is selected
    - Else if the second node exists, the second is selected
    - Else if ... third

    The option for the case where none of the nodes exist is specified by `None`.
    """

    def __init__(self, mapping: Dict[Optional[ADSGNode], ADSGNode]):
        self._mapping = mapping

    def initialize(self, sup_dsg: SupDSG, sup_choice_node: ChoiceNode, src_dsg: ADSG):
        if not isinstance(sup_choice_node, SelectionChoiceNode):
            raise SupInitializationError(self, sup_dsg, src_dsg, f'Expecting SelectionChoiceNode: {sup_choice_node!r}')

        # Check if all source nodes exist
        mapping_nodes = {key for key in self._mapping.keys() if key is not None}
        unknown_src_nodes = mapping_nodes - set(src_dsg.graph.nodes)
        if len(unknown_src_nodes) > 0:
            raise SupInitializationError(
                self, sup_dsg, src_dsg, f'Source nodes not in source DSG: {unknown_src_nodes!r}')

        # Ensure None is present
        if None not in self._mapping:
            raise SupInitializationError(
                self, sup_dsg, src_dsg, 'Case where all nodes do not exist (`None`) is not specified!')

        # Check if all target sup option nodes are available
        sup_option_nodes = sup_dsg.get_option_nodes(sup_choice_node)
        unknown_sup_opt_nodes = set(self._mapping.values()) - set(sup_option_nodes)
        if len(unknown_sup_opt_nodes) > 0:
            raise SupInitializationError(
                self, sup_dsg, src_dsg, f'Not all target nodes are choice option nodes: {unknown_sup_opt_nodes!r}')

    def resolve(self, sup_dsg: SupDSG, sup_choice_node: ChoiceNode, src_dsg: ADSG) -> SupDSG:
        assert isinstance(sup_choice_node, SelectionChoiceNode)

        # Determine selected option node
        src_nodes = {node.str_context() for node in src_dsg.graph.nodes if isinstance(node, ADSGNode)}
        sup_tgt_option_node = None
        for src_node, sup_option_node in self._mapping.items():
            if src_node is None:
                continue

            if src_node.str_context() in src_nodes:
                sup_tgt_option_node = sup_option_node
                break

        if sup_tgt_option_node is None:
            sup_tgt_option_node = self._mapping[None]

        # Apply selection choice node
        return sup_dsg.get_for_apply_selection_choice(sup_choice_node, sup_tgt_option_node)

    def __str__(self):
        mapping_str = '; '.join([f'{src_node!s} -> {tgt_node!s}' for src_node, tgt_node in self._mapping.items()])
        return f'Existence Mapping: {mapping_str}'

    def __repr__(self):
        return f'{self.__class__.__name__}(mapping={self._mapping!r})'
