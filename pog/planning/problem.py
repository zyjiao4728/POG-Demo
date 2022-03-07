# Some functions are modified from searchProblem.py - representations of search problems http://aipython.org
from pog.graph.graph import Graph
from pog.planning.searchProblem import Search_problem, Path
from pog.planning.searchNode import SearchNode
from pog.planning.action import Action, ActionType, action_seq_generator, updateGraph

import copy


class Arc():

    def __init__(self, from_node: SearchNode, to_node: SearchNode):
        self.from_node = from_node
        self.to_node = to_node
        self.action = to_node.action
        self.cost = to_node.cost

    def __repr__(self):
        """string representation of an arc"""
        return str(self.action)


# @total_ordering
class PlanningOnGraphPath(Path):
    """A path is either a node or a path followed by an arc"""

    def __repr__(self):
        """returns a string representation of a path"""
        if self.arc is None:
            return str(self.initial)
        else:
            return str(self.initial) + " ==> " + str(
                self.arc.action)  # + str(self.arc.to_node.constraints)


class PlanningOnGraphProblem(Search_problem):

    def __init__(self, start: Graph, goal: Graph, parking_place=None):
        self.start_graph = start
        self.goal_graph = goal
        self.parking_place = self.start_graph.root if parking_place is None else parking_place
        action_seq, constraints = action_seq_generator(self.start_graph,
                                                       self.goal_graph,
                                                       self.parking_place)
        self.root_search_node = SearchNode(action_seq=action_seq,
                                           constraints=constraints,
                                           current=start,
                                           goal=goal)

    def start_node(self):
        return self.root_search_node

    def is_goal(self, node: SearchNode):
        """Check if node is a goal node

        Args:
            node (Graph): node to be checked

        Returns:
            bool: True is the node is a goal. False otherwise
        """
        return node.is_leaf_node

    def neighbors(self, node: SearchNode):
        """Find neighbors of current node

        Args:
            node (Graph): current node

        Returns:
            a list of Arc: all possible neighbors
        """
        neighbors = []
        for neighbor in node.selectAction():
            (new_action_seq, constraints, new_action) = neighbor
            current = node.current.copy()

            updateGraph(current, self.goal_graph, [new_action])
            if new_action.action_type == ActionType.Place or new_action.action_type == ActionType.PicknPlace:
                (is_stable, unstable_nodes) = current.checkStability()

                if new_action.add_edge[1] in unstable_nodes:
                    continue
                elif not is_stable:
                    temp_new_action_place = Action(
                        (None, new_action.add_edge),
                        action_type=ActionType.Place)
                    new_action.add_edge = (self.start_graph.edge_dict[
                        new_action.add_edge[1]].parent_id,
                                           new_action.add_edge[1])
                    new_action.reverse = True
                    temp_new_action_pick = Action((new_action.add_edge, None),
                                                  action_type=ActionType.Pick)
                    new_action_seq.extend(
                        [temp_new_action_pick, temp_new_action_place])
                    constraints.addConstraint(temp_new_action_pick,
                                              temp_new_action_place)

                    current = node.current.copy()
                    updateGraph(current, self.start_graph, [new_action])

                    for succ in current.graph.successors(
                            new_action.add_edge[1]):
                        del_succ = Action(
                            ((new_action.add_edge[1], succ), None),
                            action_type=ActionType.Pick)
                        if del_succ in new_action_seq:
                            new_action_seq.remove(del_succ)
                            constraints.delConstraint(del_succ)
                            for action in new_action_seq:
                                if action.action_type == ActionType.Place and action.add_edge[
                                        1] == succ:
                                    add_succ = action
                                    break
                            to_ground = Action(
                                ((new_action.add_edge[1], succ),
                                 (self.parking_place, succ)),
                                action_type=ActionType.PicknPlace,
                                optimized=False)
                            from_ground = Action(
                                ((self.parking_place, succ), None),
                                action_type=ActionType.Pick,
                                optimized=True)
                            constraints.addConstraint(to_ground,
                                                      temp_new_action_pick)
                            constraints.addConstraint(temp_new_action_place,
                                                      from_ground)
                            constraints.addConstraint(from_ground, add_succ)

                            new_action_seq.extend([to_ground, from_ground])

                        else:
                            to_ground = Action(
                                ((new_action.add_edge[1], succ),
                                 (self.parking_place, succ)),
                                action_type=ActionType.PicknPlace,
                                optimized=False)
                            from_ground = Action(
                                ((self.parking_place, succ),
                                 (new_action.add_edge[1], succ)),
                                action_type=ActionType.PicknPlace,
                                optimized=False)
                            constraints.addConstraint(to_ground,
                                                      temp_new_action_pick)
                            constraints.addConstraint(temp_new_action_place,
                                                      from_ground)
                            new_action_seq.extend([to_ground, from_ground])

                if is_stable:

                    _, names = current.collision_manager.in_collision_internal(
                        return_names=True)
                    is_collision = False
                    collide_items = []
                    for pair in names:
                        if str(new_action.add_edge[1]) == pair[0]:
                            collide_items.append(int(pair[1]))
                            is_collision = True
                        elif str(new_action.add_edge[1]) == pair[1]:
                            collide_items.append(int(pair[0]))
                            is_collision = True
                        else:
                            continue

                    if is_collision:

                        old_action = copy.deepcopy(new_action)
                        new_action = Action(
                            (None,
                             (self.parking_place, old_action.add_edge[1])),
                            action_type=ActionType.Place,
                            optimized=False)
                        old_action_pick = Action(
                            ((self.parking_place, old_action.add_edge[1]),
                             None),
                            action_type=ActionType.Pick)
                        constraints.addConstraint(old_action_pick, old_action)
                        new_action_seq.extend([old_action_pick, old_action])

                        current = node.current.copy()
                        updateGraph(current, self.goal_graph, [new_action])

            constraints.delConstraint(new_action)
            to_node = SearchNode(new_action_seq, constraints, new_action,
                                 current, self.goal_graph)
            neighbors.append(Arc(node, to_node))
        return neighbors

    def heuristic(self, node: SearchNode):
        return len(node.action_sequence)
