from pog.graph.graph import Graph

import networkx as nx

from pog.graph.shape import AffordanceType


def node_match(node1, node2):
    return node1 == node2


def edge_match(edge1, edge2):
    return edge1 == edge2


def node_subst_cost(node1, node2):
    raise NotImplementedError('node_subst_cost')


def node_del_cost(node):
    raise NotImplementedError('node_del_cost')


def node_ins_cost(node):
    raise NotImplementedError('node_ins_cost')


def edge_subst_cost(edge1, edge2):
    raise NotImplementedError('edge_subst_cost')


def edge_del_cost(edge):
    raise NotImplementedError('edge_del_cost')


def edge_ins_cost(edge):
    raise NotImplementedError('edge_ins_cost')


def ged(g1, g2, with_path=False):
    if with_path:
        for node_edit_path, edge_edit_path, cost in nx.algorithms.similarity.optimize_edit_paths(
                g1.graph,
                g2.graph,
                roots=(g1.root, g2.root),
                node_match=node_match,
                edge_match=edge_match):
            if check_node_distance(g1, g2, node_edit_path):
                return cost, ((node_edit_path, edge_edit_path))
    else:  # Faster
        cost = nx.algorithms.similarity.graph_edit_distance(
            g1.graph,
            g2.graph,
            node_match=node_match,
            edge_match=edge_match,
            roots=(g1.root, g2.root))
        return cost


def ged_seq(init: Graph, goal: Graph):
    _, edit_path = ged(init, goal, with_path=True)
    print(edit_path)
    node_edit_pairs = edit_path[0]
    edge_delete_pairs = []
    edge_add_pairs = []
    for edge_pair in edit_path[1]:
        if edge_pair[0]:
            if init.edge_dict[edge_pair[0][1]].relations[
                    AffordanceType.Support]['dof'] == 'fixed':
                continue
        if edge_pair[0] == None:
            edge_add_pairs.append(edge_pair[1])
        elif edge_pair[1] == None:
            edge_delete_pairs.append(edge_pair[0])
        else:
            if init.edge_dict[edge_pair[0][1]] == goal.edge_dict[edge_pair[0]
                                                                 [1]]:
                continue
            else:
                edge_delete_pairs.append(edge_pair[0])
                edge_add_pairs.append(edge_pair[1])

    edge_edit_pairs = optimal_edge_edit_pair(init, goal, edge_delete_pairs,
                                             edge_add_pairs)

    return (node_edit_pairs, edge_delete_pairs, edge_add_pairs,
            edge_edit_pairs)


def optimal_edge_edit_pair(init, goal, edge_delete_pairs, edge_add_pairs):
    edge_edit_pairs = []
    assert len(edge_delete_pairs) == len(edge_add_pairs)
    for delete_pair in edge_delete_pairs:
        for add_pair in edge_add_pairs:
            if add_pair[1] == delete_pair[1]:
                if init.edge_dict[add_pair[1]] == goal.edge_dict[
                        delete_pair[1]]:
                    break
                edge_edit_pairs.append((delete_pair, add_pair))
                break
    return edge_edit_pairs


def check_node_distance(init: Graph, goal: Graph, node_path):
    for node_pair in node_path:
        if not node_match(init.node_dict[node_pair[0]],
                          goal.node_dict[node_pair[1]]):
            return False
    return True
