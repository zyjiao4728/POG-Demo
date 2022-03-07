from enum import Enum
from typing import List
from pog.graph.edge import Edge

from pog.graph.graph import Graph
from pog.graph.node import ContainmentState
from pog.graph.params import PairedSurface
from pog.graph.shape import AffordanceType
from pog.algorithm.structure import simulated_annealing_structure
from pog.planning.ged import ged_seq
from pog.planning.params import MAX_UPDATE_ITER

import logging
import networkx as nx


class ActionType(Enum):
    Pick = 0
    Place = 1
    PicknPlace = 2
    Open = 3
    Close = 4


class Action():

    def __init__(self,
                 edge_edit_pair,
                 action_type=None,
                 agent="",
                 optimized=True) -> None:
        if action_type is not None:
            self.action_type = action_type
        elif edge_edit_pair[0] is None and edge_edit_pair[1] is not None:
            self.action_type = ActionType.Pick
        elif edge_edit_pair[1] is None and edge_edit_pair[0] is not None:
            self.action_type = ActionType.Place
        else:
            logging.error("Please specify action type!")

        self.agent = agent
        self.del_edge = edge_edit_pair[0]
        self.add_edge = edge_edit_pair[1]
        self.optimized = optimized
        self.reverse = False

    def __eq__(self, other) -> bool:
        return hasattr(other, 'action_type') and hasattr(other, 'agent') and \
            hasattr(other, 'del_edge') and hasattr(other, 'add_edge') and \
            self.action_type == other.action_type and self.agent == other.agent and \
            self.del_edge == other.del_edge and self.add_edge == other.add_edge

    def __hash__(self) -> int:
        return hash(
            (self.action_type, self.agent, self.del_edge, self.add_edge))

    def __repr__(self) -> str:
        if self.action_type == ActionType.Pick:
            return "Pick {} from {}".format(self.del_edge[1], self.del_edge[0])
        elif self.action_type == ActionType.Place:
            return "Place {} on {}".format(self.add_edge[1], self.add_edge[0])
        elif self.action_type == ActionType.Open:
            return "Open {}".format(self.del_edge[1])
        elif self.action_type == ActionType.Close:
            return "Close {}".format(self.del_edge[1])
        elif self.action_type == ActionType.PicknPlace:
            return "PicknPlace {} to {}".format(self.del_edge[1],
                                                self.add_edge[0])
        else:
            raise NotImplementedError('self.action_type = {}'.format(
                self.action_type))

    def breakAction(self):
        assert self.del_edge is not None and self.add_edge is not None  # Can only break full actions
        assert self.action_type == ActionType.PicknPlace
        return (Action((self.del_edge, None),
                       action_type=ActionType.Pick,
                       optimized=False),
                Action((None, self.add_edge),
                       action_type=ActionType.Place,
                       optimized=False))


def updateGraph(current: Graph,
                goal: Graph,
                action_seq: List[Action],
                optimize=False):
    for action in action_seq:
        if action is None:
            continue
        elif action.action_type == ActionType.Pick:
            current.removeNode(action.del_edge[1])
        elif action.action_type == ActionType.Place:
            if action.optimized:
                current.addNode(action.add_edge[0],
                                edge=goal.edge_dict[action.add_edge[1]])
            else:
                edge = Edge(action.add_edge[0], action.add_edge[1])
                if current.root == action.add_edge[0]:
                    parent_aff_name = current.root_aff
                else:
                    parent_aff_name = PairedSurface[current.edge_dict[
                        action.add_edge[0]].relations[AffordanceType.Support]
                                                    ['child'].name]

                child_aff_name = goal.edge_dict[action.add_edge[1]].relations[
                    AffordanceType.Support]['child'].name
                edge.add_relation(
                    current.node_dict[
                        action.add_edge[0]].affordance[parent_aff_name],
                    current.node_dict[
                        action.add_edge[1]].affordance[child_aff_name],
                    dof_type='x-y',
                    pose=[0, 0, 0])
                current.addNode(action.add_edge[0], edge=edge)
                if optimize:
                    fixed_nodes = list(current.graph.nodes)
                    fixed_nodes.remove(action.add_edge[1])
                    cnt_sat = False
                    idx = 0
                    while not cnt_sat and idx < MAX_UPDATE_ITER:
                        idx += 1
                        _, cnt_sat, _ = simulated_annealing_structure(
                            current,
                            fixed_nodes=fixed_nodes,
                            random_start=True)

        elif action.action_type == ActionType.Open:
            current.node_dict[
                action.del_edge[0]].State = ContainmentState.Opened
        elif action.action_type == ActionType.Close:
            current.node_dict[
                action.del_edge[0]].State = ContainmentState.Closed
        elif action.action_type == ActionType.PicknPlace:
            if action.optimized:
                # current.removeEdge(action.del_edge[1])
                current.edge_dict[action.add_edge[1]] = goal.edge_dict[
                    action.add_edge[1]]
                current.graph.remove_edge(action.del_edge[0],
                                          action.del_edge[1])
                current.graph.add_edge(action.add_edge[0], action.add_edge[1])

            else:
                edge = Edge(action.add_edge[0], action.add_edge[1])
                if current.root == action.add_edge[0]:
                    parent_aff_name = current.root_aff
                else:
                    parent_aff_name = PairedSurface[current.edge_dict[
                        action.add_edge[0]].relations[AffordanceType.Support]
                                                    ['child'].name]

                child_aff_name = current.edge_dict[action.add_edge[
                    1]].relations[AffordanceType.Support]['child'].name
                edge.add_relation(
                    current.node_dict[
                        action.add_edge[0]].affordance[parent_aff_name],
                    current.node_dict[
                        action.add_edge[1]].affordance[child_aff_name],
                    dof_type='x-y',
                    pose=[0.5, 0.5, 0.5])

                current.edge_dict[action.add_edge[1]] = edge
                # current.graph.remove_edge(action.del_edge[0], action.del_edge[1])
                current.graph.add_edge(action.add_edge[0], action.add_edge[1])

                if optimize:
                    fixed_nodes = list(current.graph.nodes)
                    fixed_nodes.remove(action.add_edge[1])
                    simulated_annealing_structure(current,
                                                  fixed_nodes=fixed_nodes)

        else:
            logging.error("Unknown action type: {}".format(action.action_type))


class ActionConstraints():

    def __init__(self) -> None:
        self.constraints = {}

    def __repr__(self) -> str:
        return str(self.constraints)

    def violate(self, action: Action):
        return True if action in self.constraints.keys() else False

    def addConstraint(self, former, latter):
        # partial ordering
        if latter in self.constraints.keys():
            if former not in self.constraints[latter]:
                self.constraints[latter].append(former)
        else:
            self.constraints[latter] = [former]

    def delConstraint(self, action):
        delete_key = []
        for key, value in self.constraints.items():
            try:
                value.remove(action)
                if not value: delete_key.append(key)
            except ValueError:
                continue

        for key in delete_key:
            del self.constraints[key]

    def replaceConstraint(self, action1, action2):
        for _, value in self.constraints.items():
            for idx in range(len(value)):
                if value[idx] == action1:
                    value[idx] = action2
                    break


def action_seq_generator(g1: Graph, g2: Graph, parking_node):
    assert parking_node in list(g1.node_dict.keys()) and parking_node in list(
        g2.node_dict.keys())
    action_seq = []
    constraints = ActionConstraints()
    seq_from_ged = ged_seq(g1, g2)
    generate_seq_from_ged(seq_from_ged[-1], action_seq, constraints)
    generate_seq_from_accessibility(g1, g2, action_seq, constraints)

    return list(set(action_seq)), constraints


def generate_seq_from_ged(seq_from_ged, action_seq, constraints):
    for edge_pair in seq_from_ged:
        del_action = Action((edge_pair[0], None), action_type=ActionType.Pick)
        add_action = Action((None, edge_pair[1]), action_type=ActionType.Place)
        action_seq.append(del_action)
        action_seq.append(add_action)
        constraints.addConstraint(del_action, add_action)


def generate_seq_from_accessibility(g1: Graph, g2: Graph, action_seq,
                                    constraints: ActionConstraints):
    g1.updateAccessibility()
    g2.updateAccessibility()
    open_seq = []
    close_seq = []
    for action in action_seq:
        if action.action_type == ActionType.Pick:
            temp_open_seq = []
            temp_close_seq = []
            if not g1.node_dict[action.del_edge[1]].accessible:
                path_del = nx.shortest_path(g1.graph,
                                            source=g1.root,
                                            target=action.del_edge[1])
                for idx in range(len(path_del) - 1):
                    if g1.node_dict[path_del[
                            idx]].state == ContainmentState.Closed and g1.edge_dict[
                                path_del[idx + 1]].containment:
                        open_action = Action(((path_del[idx], path_del[idx]),
                                              (path_del[idx], path_del[idx])),
                                             action_type=ActionType.Open)
                        temp_open_seq.append(open_action)
                        close_action = Action(((path_del[idx], path_del[idx]),
                                               (path_del[idx], path_del[idx])),
                                              action_type=ActionType.Close)
                        temp_close_seq.append(close_action)

                for idx in range(len(temp_open_seq)):
                    if idx < len(temp_open_seq) - 1:
                        constraints.addConstraint(temp_open_seq[idx],
                                                  temp_open_seq[idx + 1])
                        constraints.addConstraint(temp_close_seq[idx + 1],
                                                  temp_close_seq[idx])
                    constraints.addConstraint(temp_open_seq[idx], action)
                    constraints.addConstraint(action, temp_close_seq[idx])

            close_seq.extend(temp_close_seq)
            open_seq.extend(temp_open_seq)

        elif action.action_type == ActionType.Place:
            temp_open_seq = []
            temp_close_seq = []
            if not g2.node_dict[action.add_edge[1]].accessible:
                path_add = nx.shortest_path(g2.graph,
                                            source=g2.root,
                                            target=action.add_edge[1])
                for idx in range(len(path_add) - 1):
                    if g2.node_dict[path_add[
                            idx]].state == ContainmentState.Closed and g2.edge_dict[
                                path_add[idx + 1]].containment:
                        open_action = Action(((path_add[idx], path_add[idx]),
                                              (path_add[idx], path_add[idx])),
                                             action_type=ActionType.Open)
                        temp_open_seq.append(open_action)
                        close_action = Action(((path_add[idx], path_add[idx]),
                                               (path_add[idx], path_add[idx])),
                                              action_type=ActionType.Close)
                        temp_close_seq.append(close_action)

                for idx in range(len(temp_open_seq)):
                    if idx < len(temp_open_seq) - 1:
                        constraints.addConstraint(temp_open_seq[idx],
                                                  temp_open_seq[idx + 1])
                        constraints.addConstraint(temp_close_seq[idx + 1],
                                                  temp_close_seq[idx])
                    constraints.addConstraint(temp_open_seq[idx], action)
                    constraints.addConstraint(action, temp_close_seq[idx])

            open_seq.extend(temp_open_seq)
            close_seq.extend(temp_close_seq)

    action_seq.extend(list(set(open_seq)))
    action_seq.extend(list(set(close_seq)))
