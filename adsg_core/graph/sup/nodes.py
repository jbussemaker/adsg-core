from adsg_core.graph.adsg_nodes import *

__all__ = ['SupNode']


class SupNode(NamedNode):
    """
    Basic node in a Supplementary DSG.

    Reference can be anything that helps the user in linking/associating the node to whatever it represents, as long as
    it is serializable/hashable and has a repr() representation.
    SupNodes with the same repr(ref) and name will be recognized as the same node.
    """

    def __init__(self, name: str, ref=None, **kwargs):
        self.ref = ref
        obj_id = self._get_obj_id(name, ref)
        super().__init__(name, obj_id=obj_id, **kwargs)

    def _get_obj_id(self, name, ref):
        return f'{name}|{ref!r}'

    def get_export_color(self) -> str:
        return '#C8E6C9'  # Green 100
