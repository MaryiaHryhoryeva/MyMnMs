from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import List, FrozenSet, Dict, Set, Optional, Tuple, Union

from mnms.tools.cost import create_link_costs
from mnms.log import create_logger

log = create_logger(__name__)


# class GraphElement(ABC):
#     """Base class for the creation of a graph element

#     Parameters
#     ----------
#     id: str
#         Id of the element

#     """
#     __slots__ = ('id',)

#     def __init__(self, id: str):
#         self.id = id

#     @abstractmethod
#     def __load__(cls, data: dict):
#         pass

#     @abstractmethod
#     def __dump__(self) -> dict:
#         pass

#     @abstractmethod
#     def __deepcopy__(self, memodict={}):
#         pass


# class Node(GraphElement):
#     """Class representing a topological node, it can refer to a GeoNode id

#     Parameters
#     ----------
#     id: str
#         The identifier for this TopoNode
#     ref_node: str
#         A reference to GeoNode (default is None)

#     """
#     __slots__ = ('adj',
#                  'radj',
#                  'position',
#                  'reference_node',
#                  'layer',
#                  '_exclude_movements')
                 'links')

#     def __init__(self, id: str, layer: str, ref_node: str, exclude_movements: Optional[Dict[str, Set[str]]] = None):
#         super(Node, self).__init__(id)

#         self.adj = set()
#         self.radj = set()

#         self.reference_node = ref_node
#         self.position = None
#         self.layer = layer
#         self._exclude_movements = dict() if exclude_movements is None else exclude_movements
#         self._exclude_movements[None] = set()
        self.links = set()

#     def get_exits(self, predecessor: Optional[str] = None):
#         return (i for i in self.adj if predecessor not in self._exclude_movements or i not in self._exclude_movements[predecessor])

#     def get_entrances(self, predecessor: Optional[str] = None):
#         return (i for i in self.radj if i not in self._exclude_movements or predecessor not in self._exclude_movements[i])

#     def __repr__(self):
#         return f"Node({self.id.__repr__()})"

#     @classmethod
#     def __load__(cls, data: dict) -> "TopoNode":
#         exclude_movements = data.get('EXCLUDE_MOVEMENTS', dict())
#         return cls(data['ID'], data['LAYER'], data['REF_NODE'], {key: set(val) for key, val in exclude_movements.items()})

#     def __dump__(self) -> dict:
#         return {'ID': self.id,
#                 'REF_NODE': self.reference_node,
#                 'LAYER': self.layer,
#                 'EXCLUDE_MOVEMENTS': {key: list(val) for key, val in self._exclude_movements.items() if key is not None}}

#     def __deepcopy__(self, memodict={}):
#         cls = self.__class__
#         result = cls(self.id, self.layer, self.reference_node)
#         return result


# class ConnectionLink(GraphElement):
    # __slots__ = ('upstream', 'downstream', 'costs', 'reference_links', 'layer')

    # def __init__(self, lid, upstream_node, downstream_node, costs, reference_links, layer=None):
    #     super(ConnectionLink, self).__init__(lid)
    #     self.upstream = upstream_node
    #     self.downstream = downstream_node
    #     self.costs: Dict = create_link_costs()

    #     if costs is not None:
    #         self.costs.update(costs)
    #     self.reference_links = reference_links if reference_links is not None else []
    #     self.layer = layer

    # def __repr__(self):
    #     return f"ConnectionLink(id={self.id.__repr__()}, upstream={self.upstream.__repr__()}, downstream={self.downstream.__repr__()})"

    # @classmethod
    # def __load__(cls, data: dict) -> "ConnectionLink":
    #     return cls(data['ID'], data['UPSTREAM'], data['DOWNSTREAM'], data['COSTS'], data['REF_LINKS'], data['LAYER'])

    # def __dump__(self) -> dict:
    #     return {'ID': self.id,
    #             'UPSTREAM': self.upstream,
    #             'DOWNSTREAM': self.downstream,
    #             'COSTS': {key: val for key, val in self.costs.items() if key not in ("_default", 'speed')},
    #             'REF_LINKS': self.reference_links,
    #             'LAYER': self.layer}

    # def __deepcopy__(self, memodict={}):
    #     cls = self.__class__
    #     result = cls(self.id,
    #                  self.upstream,
    #                  self.downstream,
    #                  deepcopy(self.costs),
    #                  deepcopy(self.reference_links),
    #                  self.layer)
    #     return result


# class TransitLink(GraphElement):
#     """ Link between two different mobility service

#     Parameters
#     ----------
#     lid: str
#         id of the link
#     upstream_node: str
#         id of upstream node
#     downstream_node: str
#         id of downstream node
#     costs: dict
#         dictionary of costs
#     """
#     __slots__ = ('upstream', 'downstream', 'costs')

#     def __init__(self, lid, upstream_node, downstream_node, costs=None):
#         super(TransitLink, self).__init__(lid)
#         self.upstream = upstream_node
#         self.downstream = downstream_node
#         self.costs: Dict = create_link_costs()
#         if costs is not None:
#             self.costs.update(costs)

#     def __repr__(self):
#         return f"TransitLink(id={self.id.__repr__()}, upstream={self.upstream.__repr__()}, downstream={self.downstream.__repr__()})"

#     @classmethod
#     def __load__(cls, data: dict) -> "TransitLink":
#         return cls(data['ID'], data['UPSTREAM'], data['DOWNSTREAM'], data['COSTS'])

#     def __dump__(self) -> dict:
#         return {'ID': self.id,
#                 'UPSTREAM': self.upstream,
#                 'DOWNSTREAM': self.downstream,
#                 'COSTS': {key: val for key, val in self.costs.items() if key not in ("_default", 'speed')}}

#     def __deepcopy__(self, memodict={}):
#         cls = self.__class__
#         result = cls(self.id,
#                      self.upstream,
#                      self.downstream,
#                      deepcopy(self.costs))
#         return result


class Zone():
    """Set of sections that define a geographic zone

    Parameters
    ----------
    resid: str
        id of the zone
    links: list
        list of sections id
    mobility_services:
        list of mobility services present in the zone
    """
    __slots__ = ('mobility_services', 'links')

    def __init__(self, resid: str, links:List[str]=[], mobility_services:List[str]=[]):
        self.id = resid
        self.mobility_services = frozenset(mobility_services)
        self.links: FrozenSet[str] = frozenset(links)

    def __dump__(self) -> dict:
        return {'ID': self.id, 'MOBILITY_SERVICES': list(self.mobility_services), 'LINKS': list(self.links)}

    @classmethod
    def __load__(cls, data: dict):
        return Zone(data['ID'], data['LINKS'], data['MOBILITY_SERVICES'])

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls(self.id,
                     deepcopy(self.links))
        return result


# class OrientedGraph(object):
#     """Basic class for an oriented graph.

#     Attributes
#     ----------
#     nodes : dict
#         Dict of nodes, key is the id of node, value is either a GeoNode or a Toponode
#     links : type
#         Dict of sections, key is the tuple of nodes, value is either a GeoLink or a TopoLink

#     """

#     __slots__ = ('nodes', 'links', 'node_referencing', '_map_lid_nodes')

#     def __init__(self):
#         self.nodes: Dict[str, Node] = dict()
#         self.links: Dict[Tuple[str, str], Union[TransitLink, ConnectionLink]] = dict()
#         self._map_lid_nodes: Dict[str, Tuple[str, str]] = dict()

#         self.node_referencing = defaultdict(list)

#     @property
#     def nb_nodes(self):
#         return len(self.nodes)

#     @property
#     def nb_links(self):
#         return len(self.links)

#     def get_link(self, id:str):
#         return self.links[self._map_lid_nodes[id]]

#     def add_node(self, node:Node):
#         if node.reference_node is not None:
#             self.node_referencing[node.reference_node].append(node.id)
#         self.nodes[node.id] = node

#     def add_link(self, link: Union[TransitLink, ConnectionLink]):
#         self.links[(link.upstream, link.downstream)] = link
#         self._map_lid_nodes[link.id] = (link.upstream, link.downstream)

#         unode = self.nodes[link.upstream]
#         dnode = self.nodes[link.downstream]

#         unode.adj.add(link.downstream)
#         dnode.radj.add(link.upstream)

        unode.links.add(link)

#     def create_node(self,
#                     nid: str,
#                     mobility_service:str,
#                     ref_node: str,
#                     exclude_movements: Optional[Dict[str, Set[str]]] = None) -> None:
#         assert nid not in self.nodes, f"Node '{nid}' already in graph"
#         new_node = Node(nid, mobility_service, ref_node, exclude_movements)
#         self.add_node(new_node)

#     def create_link(self,
#                     lid:str,
#                     upstream_node: str,
#                     downstream_node: str,
#                     costs: Dict[str, float],
#                     reference_links: List[str],
#                     mobility_service: str = None) -> None:
#         assert (upstream_node, downstream_node) not in self.links, f"Nodes {upstream_node}, {downstream_node} already connected"
#         assert lid not in self._map_lid_nodes, f"Link id {lid} already exist"

#         link = ConnectionLink(lid, upstream_node, downstream_node, costs, reference_links, mobility_service)
#         self.add_link(link)
