from typing import List
from pog.graph.edge import Edge
from pog.graph.graph import Graph
from pog.graph.params import PairedSurface
from pog.graph.shape import AffordanceType
from pog.graph.node import ContainmentState
from pog.planning.planner import PlanningOnGraphPath
from pog.planning.action import Action, updateGraph 

import vedo
import networkx as nx

def operation_picknplace(node, source_object, target_object): # for foward astar search
    if not node.node_dict[source_object].accessible:
        return []
    elif node.node_dict[source_object].state is not None:
        new_node = node.getSubGraph(node.root)
        if new_node.node_dict[source_object].state == ContainmentState.Closed:
            new_node.node_dict[source_object].state = ContainmentState.Opened
            new_node.updateAccessibility()
        else:
            new_node.node_dict[source_object].state = ContainmentState.Closed
            new_node.updateAccessibility()
        return [new_node]
    elif source_object == node.root or node.edge_dict[source_object].relations[AffordanceType.Support]['dof'] == 'fixed' or \
        source_object == target_object or not node.node_dict[target_object].accessible:
        return []
    else:
        try:
            nx.shortest_path_length(node.graph, source_object, target_object)
            return []
        except nx.exception.NetworkXNoPath:
            if target_object == node.root:
                target_parent_aff = node.root_aff
            else:
                try:
                    target_parent_aff = PairedSurface[node.edge_dict[target_object].relations[AffordanceType.Support]['child'].name]
                except KeyError:
                    return []
            if bool(node.graph.succ[source_object]):
                source_child_aff = node.edge_dict[source_object].relations[AffordanceType.Support]['child'].name
                new_node = node.getSubGraph(node.root)
                new_edge = Edge(parent=target_object, child=source_object)
                new_edge.add_relation(node.node_dict[target_object].affordance[target_parent_aff], node.node_dict[source_object].affordance[source_child_aff], dof_type='x-y', pose=[0,0,0])
                new_node.edge_dict[source_object] = new_edge
                new_node.graph.add_edge(new_edge.parent_id, new_edge.child_id, edge=new_edge)
                return [new_node]
            else:
                new_node_lists = []
                for possible_source_child_aff in node.node_dict[source_object].affordance.keys():
                    new_node = node.getSubGraph(node.root)
                    new_edge = Edge(parent=target_object, child=source_object)
                    new_edge.add_relation(node.node_dict[target_object].affordance[target_parent_aff], node.node_dict[source_object].affordance[possible_source_child_aff], dof_type='x-y', pose=[0,0,0])
                    new_node.edge_dict[source_object] = new_edge
                    new_node.graph.add_edge(new_edge.parent_id, new_edge.child_id, edge=new_edge)
                    new_node_lists.append(new_node)
                return new_node_lists    


def path_to_action_sequence(path : PlanningOnGraphPath):
    action_sequence = []
    for node in path.nodes():
        action_sequence.append(node.action)
    return reversed(action_sequence)

def apply_action_sequence_to_graph(init : Graph, goal : Graph, action_sequence : List[Action], visualize=False, save_step=False):
    current = init.copy()
    success = True
    idx = 0
    for action in action_sequence:
        if save_step:
            current.toJson(file_dir="result/", file_name="{}.json".format(idx))
            idx += 1
        if action and action.reverse:
            updateGraph(current, init, [action], optimize=True)
        else:
            updateGraph(current, goal, [action], optimize=True)
        is_collision, names = current.collision_manager.in_collision_internal(return_names=True)
        print(action, current.checkStability(), (is_collision, names))
        success = success and (current.checkStability() and not is_collision)
        if visualize:
            current.create_scene()
            if action:
                vedo.show(current.scene.dump(concatenate=True), axes=0, resetcam=False, interactive=False)
            else:
                vedo.show(current.scene.dump(concatenate=True), axes=0,camera={'pos':(0,-3,3),}, size=(2000,2000), new=True, azimuth=0, elevation=-45, roll=0)
            input('Press [Enter] to continue.')

    return current, success