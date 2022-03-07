import copy
from math import inf
import numpy as np
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_tree
from functools import total_ordering

from pog.graph import node, edge
from pog.graph.shape import AffordanceType
from pog.graph.utils import *
from pog.graph.graph import Graph

import logging
import random

from pog.graph.params import FRICTION_ANGLE_THRESH, MAX_INITIAL_TRIES, PairedSurface, ContainmentSurface
from pog.graph.utils import match


class NodeLite():

    def __init__(self, node: node.Node) -> None:
        """A light-weight node class. Remove SDF and shape, 
        which is unnecessary for genetic programming

        Args:
            node (node.Node): Original node object
        """
        self.id = node.id
        self.affordance = {}
        for aff_name, aff in node.affordance.items():
            self.affordance[aff_name] = aff.to_lite()

    def __repr__(self) -> str:
        return str(self.id)

    def __eq__(self, other) -> bool:
        return self.id == other.id  # NOTE: we are assuming object has unique id


class Gene():

    def __init__(self, parent: dict, child: dict) -> None:
        """Gene class

        Args:
            parent (dict): parent affordance
            child (dict): child affordance
        """
        self.parent_id = parent['node_id']
        self.child_id = child['node_id']
        self.parent_affordance_name = parent['name']
        self.child_affordance_name = child['name']

    def __repr__(self) -> str:
        return 'Gene: {} ---> {}: {} ---> {} \n'.format(
            self.parent_id, self.child_id, self.parent_affordance_name,
            self.child_affordance_name)

    def __eq__(self, other) -> bool:
        return self.parent_id == other.parent_id and self.child_id == other.child_id and self.parent_affordance_name == other.parent_affordance_name and self.child_affordance_name == other.child_affordance_name


@total_ordering
class Chromosome():

    def __init__(self,
                 node_dict_lite: dict = None,
                 node_dict: dict = None,
                 root=None,
                 root_aff=None,
                 constraints={},
                 initchromo=None,
                 **kwargs) -> None:
        """Chromosome class and operations. A light version for graph, but remove nested functions

        Args:
            node_dict (dict): dictionary of NodeLite objects
            root (int, optional): root node id. Defaults to None.
            root_aff (str, optional): root affordance name (for supporting). Defaults to None.
            constraints (dict, optional): Constraints on chromosome. Defaults to {}.
        """
        self.node_dict = {}
        if node_dict_lite is not None:
            self.node_dict = node_dict_lite
        elif node_dict is not None:
            for key, value in node_dict.items():
                self.node_dict[key] = NodeLite(value)
        else:
            logging.error(
                'Unable to initialize chromosome. Please specify node_dict or node_dict_lite'
            )

        self.num_nodes = len(self.node_dict.keys())
        self.num_edges = self.num_nodes - 1
        if self.num_nodes < 2:
            logging.error('At least 2 nodes is required.')

        self.initialized = False
        self.fitness = None
        self.cnt_sat = False
        self.chromosome = {}
        self.initchromo = {}
        self.chromograph = nx.DiGraph()
        self.root = root
        self.root_aff = root_aff
        self.vertical_dir = np.array([0., 0., 1.])
        self.global_transform = {}
        self.depth_dict = {}
        self.history = []
        self.history_str = []
        self.num_generation = 0

        self.height_cnt = constraints[
            'height'] if 'height' in constraints.keys() else inf  # float
        self.gene_cnt = constraints['gene'] if 'gene' in constraints.keys(
        ) else {}  # partial chromo
        self.prop_cnt = constraints[
            'propagation'] if 'propagation' in constraints.keys() else {
            }  # edge child-parent
        self.cont_cnt = constraints[
            'contain'] if 'contain' in constraints.keys() else {
            }  # edge child-parent
        self.fixed_nodes = constraints['fixed'] if 'fixed' in constraints.keys(
        ) else {}  # nodes that fixed wrt their parents
        self.no_crossoverto_nodes = constraints[
            'nocrossoverto'] if 'nocrossoverto' in constraints.keys() else []

        # Define constraints
        # gene_cnt: no crossover, can mutate, pose can change
        # prop_cnt: child must along the branch of parent
        # height_cnt: total height constraint
        # fixed_nodes: no crossover, no mutation, pose cannot change
        self.no_crossover_nodes = list(self.gene_cnt.keys())
        self.no_crossover_nodes.extend(list(self.fixed_nodes.keys()))
        self.no_mutation_nodes = []  #list(self.gene_cnt.keys())
        self.no_mutation_nodes.extend(list(self.fixed_nodes.keys()))

        if initchromo is not None:
            self.initchromo = copy.deepcopy(initchromo)
            self.chromosome = copy.deepcopy(initchromo)
            self.toNXGraph()
            if not self.checkStability():
                logging.error(
                    'Unstable configuration for chromosome: {}'.format(
                        self.chromosome))
            self.initialized = True

    def initialize(self):
        count = 0
        while not self.initialized and count < MAX_INITIAL_TRIES:
            self.initialized = self.initChromosome()
            count += 1
        if not self.initialized:
            logging.warning(
                'Unable to initialize chromosome within given time.')

        self.initchromo = copy.deepcopy(self.chromosome)
        return self.initialized

    def __repr__(self) -> str:
        return str(self.chromosome.values())

    def __eq__(self, other) -> bool:
        return self.fitness == other.fitness

    def __lt__(self, other) -> bool:
        return self.fitness < other.fitness

    def toNXGraph(self):
        """Convert chromosome to networkx graph
        """
        self.chromograph.clear()

        for _, node in self.node_dict.items():
            self.chromograph.add_node(node.id, node=node)

        for edge in self.chromosome.values():
            self.chromograph.add_edge(edge.parent_id, edge.child_id, edge=edge)

        assert nx.algorithms.tree.recognition.is_tree(self.chromograph)

    def initChromosome(self):
        """Initialize chromosome

        Args:
            root (int, optional): root id. Defaults to None.
            root_aff (str, optional): root affordance. Defaults to None.

        Returns:
            bool: If Ture. A feasible initialization found.
        """

        if self.root is None:
            candidate_root = list(self.node_dict.keys())
            random.shuffle(candidate_root)
            self.root = candidate_root.pop()
            while candidate_root and self.root in self.gene_cnt.keys():
                self.root = candidate_root.pop()
        if isinstance(self.root_aff, str):
            assert self.root_aff in self.node_dict[self.root].affordance.keys()
            aff = self.node_dict[self.root].affordance[self.root_aff]
            aff_name = self.root_aff
        else:
            aff_name, aff = random.choice(
                list(self.node_dict[self.root].affordance.items()))

        self.vertical_dir = aff['transform'][0:3, 2]

        if aff_name in PairedSurface.keys():  # support up
            for key, value in self.node_dict.items():
                if key != self.root and key in self.fixed_nodes.keys():
                    self.chromosome[key] = self.fixed_nodes[key]
                elif key != self.root and key in self.gene_cnt.keys():
                    self.chromosome[key] = self.gene_cnt[key]
                elif key != self.root and key not in self.gene_cnt.keys():
                    self.chromosome[key] = Gene(
                        parent=aff,
                        child=random.choice(list(value.affordance.values())))
            self.toNXGraph()
            if self.checkStability():
                return True

        return False

    @staticmethod
    def is_identical(self, other) -> bool:
        return self.chromosome.values() == other.chromosome.values()

    #TODO: Add pick and place operation for planning

    def mutate(self):
        """Mutate operation
        """
        assert len(list(self.chromosome.values())) == self.num_edges

        candidate_mutate_node_id = list(self.node_dict.keys())
        random.shuffle(candidate_mutate_node_id)
        mutate_node_id = candidate_mutate_node_id.pop()

        while mutate_node_id in self.no_mutation_nodes:
            if not candidate_mutate_node_id:
                return  # no possible mutation
            mutate_node_id = candidate_mutate_node_id.pop()

        mutate_node = self.node_dict[mutate_node_id]
        if len(mutate_node.affordance.items()) <= 1:
            return

        aff_name, _ = random.choice(list(mutate_node.affordance.items()))

        if aff_name in PairedSurface.keys():  #support down
            if mutate_node_id == self.root:
                self.vertical_dir = mutate_node.affordance[aff_name][
                    'transform'][0:3, 2]  # vertical axes
                self.root_aff = aff_name
            for key, value in self.chromosome.items():
                if value.child_id == mutate_node_id:
                    value.child_affordance_name = PairedSurface[aff_name]
                elif value.parent_id == mutate_node_id:
                    value.parent_affordance_name = aff_name
        else:
            for key, value in self.chromosome.items():
                if key == mutate_node_id:
                    value.child_affordance_name = aff_name
        try:
            self.history_str.append('Mutation: {}: {} --> {}'.format(
                mutate_node_id,
                self.chromosome[mutate_node_id].child_affordance_name,
                aff_name))
        except KeyError:
            self.history_str.append('Mutation: {}: {} --> {}'.format(
                mutate_node_id, self.root_aff, aff_name))
        return

    def crossover(self):
        """Crossover operation
        """
        assert len(list(self.chromosome.values())) == self.num_edges

        self.num_generation += 1

        candidate_crossover_nodes = list(self.node_dict.keys())
        random.shuffle(candidate_crossover_nodes)
        crossover_node_id = candidate_crossover_nodes.pop()

        while crossover_node_id in self.no_crossover_nodes or crossover_node_id == self.root:
            if not candidate_crossover_nodes:
                return  # cannot perform crossover
            else:
                crossover_node_id = candidate_crossover_nodes.pop()

        old_parent_node_id = self.chromosome[crossover_node_id].parent_id
        old_parent_aff_name = self.chromosome[
            crossover_node_id].parent_affordance_name

        candidate_new_parent_node_id = list(self.node_dict.keys())
        random.shuffle(candidate_new_parent_node_id)
        self.computeGlobalTF()
        while candidate_new_parent_node_id:
            new_parent_node_id = candidate_new_parent_node_id.pop()
            if new_parent_node_id in self.no_crossoverto_nodes: continue
            new_parent_node = self.node_dict[new_parent_node_id]
            # if crossover_node_id == new_parent_node_id: # NOTE: Comment to allow same parent but different affordance for crossover
            #     continue
            try:  # avoid cyclic in graph
                nx.shortest_path_length(self.chromograph, crossover_node_id,
                                        new_parent_node_id)
                continue
            except nx.exception.NetworkXNoPath:
                candidate_new_parent_aff_name = list(
                    new_parent_node.affordance.keys())
                random.shuffle(candidate_new_parent_aff_name)
                while candidate_new_parent_aff_name:
                    new_parent_aff_name = candidate_new_parent_aff_name.pop()
                    new_parent_aff = new_parent_node.affordance[
                        new_parent_aff_name]
                    if new_parent_aff_name in PairedSurface.keys(
                    ):  # support up
                        tf = self.global_transform[
                            new_parent_node_id] @ new_parent_aff['transform']
                        uv1 = tf[0:3, 2] / np.linalg.norm(tf[0:3, 2])
                        uv2 = self.vertical_dir / np.linalg.norm(
                            self.vertical_dir)
                        angle = np.arccos(np.dot(uv1, uv2))
                        if angle < FRICTION_ANGLE_THRESH:
                            self.chromosome[
                                crossover_node_id].parent_affordance_name = new_parent_aff_name
                            self.chromosome[
                                crossover_node_id].parent_id = new_parent_node_id
                            self.toNXGraph()
                            self.history_str.append(
                                'Crossover: {}: {} --> {}: {} --> {}'.format(
                                    crossover_node_id, old_parent_node_id,
                                    new_parent_node_id, old_parent_aff_name,
                                    new_parent_aff_name))
                            return
        return

    def checkPropConstraints(self):
        """Check if all propagation constraints are satisfied

        Returns:
            satisfy (bool): True if all constraints are satisfied
            cost (int): Number of violations
        """
        satisfy = True
        cost = 0
        for child, parent in self.prop_cnt.items():
            try:
                nx.shortest_path_length(self.chromograph, parent, child)
            except nx.exception.NetworkXNoPath:
                satisfy = False
                cost += 1
        return satisfy, cost

    def checkContConstraints(self):
        """Check if all containment constraints are satisfied

        Returns:
            satisfy (bool): True if all constraints are satisfied
            cost (int): Number of violations
        """
        satisfy = True
        cost = 0
        for child, parent in self.cont_cnt.items():
            try:
                shortest_path = nx.shortest_path(self.chromograph, parent,
                                                 child)
                parent_affordance_name = self.chromosome[
                    shortest_path[1]].parent_affordance_name
                if not self.node_dict[shortest_path[0]].affordance[
                        parent_affordance_name]['containment']:
                    satisfy = False
                    cost += 1
            except nx.exception.NetworkXNoPath:
                satisfy = False
                cost += 1
        return satisfy, cost

    def checkHeightConstraints(self):
        if self.height_cnt == inf:
            total_height_cnt = True
            total_height_cost = 0
        else:
            total_height_cnt, total_height_cost = self.checkSubHeightConstraints(
                self.root, self.height_cnt)

        height_cnt = True
        height_cost = 0
        for key, value in self.chromosome.items():
            if value.parent_affordance_name in ContainmentSurface:
                aff_height_cnt = self.node_dict[value.parent_id].affordance[
                    value.parent_affordance_name]['height']
                # print(aff_height_cnt)
                temp_height_cnt, temp_height_cost = self.checkSubHeightConstraints(
                    key, aff_height_cnt)
                height_cnt = height_cnt and temp_height_cnt
                height_cost += temp_height_cost

        return total_height_cnt and height_cnt, 0 #height_cost + total_height_cost

    def checkSubHeightConstraints(self, node_id, height_cnt):
        """Check if height constraint is satisfied

        Returns:
            (bool): True if height constraint is satisfied
            (float): Cost of violations
        """
        if node_id == self.root:
            max_height = 0
            self.computeGlobalTF()
            for node, tf in self.global_transform.items():
                if node != self.root:
                    v1 = tf[0:3, 3]
                    v2 = self.vertical_dir
                    height = np.linalg.norm(
                        np.dot(v1, v2) / np.dot(v2, v2) *
                        v2) + self.node_dict[node].affordance[self.chromosome[
                            node].child_affordance_name]['height'] / 2.0
                else:
                    v1 = tf[0:3, 3]
                    v2 = self.vertical_dir
                    height = np.linalg.norm(
                        np.dot(v1, v2) / np.dot(v2, v2) * v2) * 2.0

                if height > max_height:
                    max_height = height

        else:
            max_height = 0
            subtree_at_node_id = dfs_tree(self.chromograph, node_id)
            self.computeGlobalTF()
            for node in subtree_at_node_id.nodes():
                # if node != node_id:
                v1 = self.global_transform[node][0:3, 3]
                v2 = self.vertical_dir
                height = np.linalg.norm(
                    np.dot(v1, v2) / np.dot(v2, v2) *
                    v2) + self.node_dict[node].affordance[self.chromosome[
                        node].child_affordance_name]['height'] / 2.0

                if height > max_height:
                    max_height = height

            v1 = self.global_transform[node_id][0:3, 3]
            v2 = self.vertical_dir
            base_height = np.linalg.norm(
                np.dot(v1, v2) / np.dot(v2, v2) *
                v2) - self.node_dict[node_id].affordance[self.chromosome[
                    node_id].child_affordance_name]['height'] / 2.0
            # print(max_height, base_height)
            max_height = max_height - base_height

        return max_height < height_cnt, max_height - height_cnt

    def checkStability(self):
        """Check if self is stable

        Returns:
            (bool): True if self is stable
        """
        stable = True
        self.computeGlobalTF()
        for node_id in self.chromosome.keys():
            parent_id = self.chromosome[node_id].parent_id
            parent_aff_name = self.chromosome[node_id].parent_affordance_name
            tf = self.global_transform[parent_id] @ self.node_dict[
                parent_id].affordance[parent_aff_name]['transform']
            uv1 = tf[0:3, 2] / np.linalg.norm(tf[0:3, 2])
            uv2 = self.vertical_dir / np.linalg.norm(self.vertical_dir)
            angle = np.arccos(np.dot(uv1, uv2))
            if angle > FRICTION_ANGLE_THRESH:
                stable = False
        return stable

    def trackDepth(self):
        """Find nodes at each depth and store it in self.depth_dict
        """
        max_depth = len(nx.algorithms.dag_longest_path(self.chromograph))
        node_depth = nx.shortest_path_length(self.chromograph, self.root)
        self.depth_dict = {}
        for depth in range(0, max_depth):
            temp_depth_list = []
            for key, value in node_depth.items():
                if value == depth:
                    temp_depth_list.append(key)
            self.depth_dict[depth] = temp_depth_list

    def computeGlobalTF(self):
        """Compute transformations from root to all nodes in chromosome 
        """
        max_depth = len(nx.algorithms.dag_longest_path(self.chromograph))
        node_depth = nx.shortest_path_length(self.chromograph, self.root)
        self.global_transform = {}
        self.global_transform[self.root] = np.identity(4)
        for i in range(1, max_depth):
            for key, value in node_depth.items():
                if value == i:
                    tf = np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0),
                                   (0, 0, 0, 1)))
                    tf = self.__genGlobalTFHelper(key, value, tf)
                    try:
                        self.global_transform[key] = tf
                    except KeyError:
                        logging.error(
                            'Cannot find edge of child ID {} in edge list.'.
                            format(key))

    def __genGlobalTFHelper(self, key, value, tf):
        if value == 0:
            return tf
        elif value > 0:
            value -= 1
            parent_id = self.chromosome[key].parent_id
            parent_aff_name = self.chromosome[key].parent_affordance_name
            child_aff_name = self.chromosome[key].child_affordance_name
            tf = np.dot(p2cTF(self.node_dict[parent_id].affordance[parent_aff_name]['transform'], \
                self.node_dict[key].affordance[child_aff_name]['transform'], affTFxy([0.,0.,0.])), tf)
            return self.__genGlobalTFHelper(parent_id, value, tf)


def to_graph(chromo: Chromosome, node_dict, g: Graph = None) -> Graph:
    """Convert chromosome to graph

    Args:
        chromo (Chromosome): chromosome to be converted
        node_dict (dict): dictionary of nodes
    """
    if g is not None:

        def fn():
            edge_dict = {}

            for item in chromo.chromosome.values():
                if item.child_id not in chromo.fixed_nodes:
                    edge_dict[item.child_id] = edge.Edge(parent=item.parent_id,
                                                         child=item.child_id)
                    edge_dict[item.child_id].add_relation(node_dict[item.parent_id].affordance[item.parent_affordance_name],\
                        node_dict[item.child_id].affordance[item.child_affordance_name], pose=[0.,0.,0.])
                else:
                    edge_dict[item.child_id] = edge.Edge(parent=item.parent_id,
                                                         child=item.child_id)
                    edge_dict[item.child_id].add_relation(node_dict[item.parent_id].affordance[item.parent_affordance_name],\
                        node_dict[item.child_id].affordance[item.child_affordance_name], \
                            dof_type = 'fixed', pose=g.edge_dict[item.child_id].relations[AffordanceType.Support]['pose'])
            root_id = chromo.root
            return node_dict, edge_dict, root_id
    else:

        def fn():
            edge_dict = {}

            for item in chromo.chromosome.values():
                edge_dict[item.child_id] = edge.Edge(parent=item.parent_id,
                                                     child=item.child_id)
                edge_dict[item.child_id].add_relation(node_dict[item.parent_id].affordance[item.parent_affordance_name],\
                        node_dict[item.child_id].affordance[item.child_affordance_name], pose=[0.,0.,0.])

            root_id = chromo.root
            return node_dict, edge_dict, root_id

    return Graph('Graph of choromosome {}'.format(str(chromo)), fn=fn)


def toChromosome(g: Graph):
    node_dict_lite = {}
    constraints = {}
    parent_aff_name = None
    for key, value in g.node_dict.items():
        node_dict_lite[key] = NodeLite(value)

    constraints = {}
    fixed_nodes = {}
    initchromo = {}
    for key, value in g.edge_dict.items():
        if value.relations[AffordanceType.Support]['dof'] == 'fixed':
            fixed_nodes[key] = Gene(
                {
                    'node_id': value.parent_id,
                    'name':
                    value.relations[AffordanceType.Support]['parent'].name
                }, {
                    'node_id': value.child_id,
                    'name':
                    value.relations[AffordanceType.Support]['child'].name
                })

        if g.edge_dict[key].parent_id == g.root and parent_aff_name is None:
            parent_aff_name = g.edge_dict[key].relations[
                AffordanceType.Support]['parent'].name

        initchromo[key] = Gene(
            {
                'node_id': value.parent_id,
                'name': value.relations[AffordanceType.Support]['parent'].name
            }, {
                'node_id': value.child_id,
                'name': value.relations[AffordanceType.Support]['child'].name
            })

    fixed_nodes[g.root] = Gene({
        'node_id': g.root,
        'name': 'root'
    }, {
        'node_id': g.root,
        'name': 'root'
    })

    constraints['nocrossoverto'] = [g.root]

    constraints['fixed'] = fixed_nodes

    return Chromosome(node_dict_lite=node_dict_lite,
                      root=g.root,
                      root_aff=parent_aff_name,
                      constraints=constraints,
                      initchromo=initchromo)

