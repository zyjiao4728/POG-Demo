from typing import List
import networkx as nx

from pog.algorithm.utils import *
from pog.graph.graph import Graph
from pog.algorithm.annealing import simulated_annealing
from pog.algorithm.gradient import gradient_descent

import time, logging


def simulated_annealing_structure(g: Graph,
                                  fixed_nodes=[],
                                  reverse=False,
                                  random_start=False,
                                  verbose=False,
                                  visualize_step=False,
                                  **kwargs):
    """Simulated annealing algorithm for stability optimization on a structured scene graph

    Args:
        g (Graph): scene graph
        fixed_nodes (list, optional): a list of fixed nodes, which are not going to be optimized. Defaults to [].
        reverse (bool, optional): optimize scene graph in reverse order (from leaf to root). Defaults to False.
        random_start (bool, optional): Randomly select initial configuration. Defaults to False.
        verbose (bool, optional): More outputs. Defaults to False.
        visualize_step (bool, optional): visualize optimization results in everystep. Defaults to False.

    Returns:
        g (Graph): optimized scene graph
        (bool): If true, all constraints are satisfied
        best_eval_total (float): total cost of optimization result
    """
    try:
        if kwargs['optim_method'] == 'gradient':
            fn = gradient_descent
        elif kwargs['optim_method'] == 'annealing':
            fn = simulated_annealing
        else:
            fn = gradient_descent
    except KeyError:
        fn = gradient_descent

    pose_history: dict[int, List[dict]] = {}
    node_depth = nx.shortest_path_length(g.graph, g.root)
    max_depth = max(node_depth.values())
    i = 0
    best_eval_total = 0
    depth_range = range(max_depth)
    if reverse:
        depth_range = reversed(range(max_depth))
    for depth in depth_range:
        for node_id in g.depth_dict[depth]:
            idx = []
            for succ in g.graph.successors(node_id):
                if succ not in fixed_nodes:
                    idx.append(succ)
            if len(idx) > 0:
                i += 1
                if verbose:
                    logging.debug('Problem: {}, idx: {}'.format(i, idx))
                sg = g.getSubGraph(node_id) if reverse else g.getSubGraph(
                    g.root, depth=depth + 1)
                pose = sg.getPose(edge_id=idx)
                object_pose_dict, _ = gen_bound(pose)
                start = time.time()
                best, best_eval = fn(
                    [objective],
                    sg,
                    node_id=idx,
                    random_start=random_start,
                    verbose=verbose,
                )
                end = time.time()
                if verbose:
                    logging.debug(
                        'Finished optimization in {:.4f} seconds'.format(
                            end - start))
                new_pose = arr2pose(best, object_pose_dict, pose)
                g.setPose(new_pose)
                if visualize_step:
                    sg.setPose(new_pose)
                    sg.create_scene()
                    sg.scene.show()
                del sg
                best_eval_total += best_eval
    pose = g.getPose()
    object_pose_dict, _ = gen_bound(pose)
    return g, checkConstraints(pose2arr(pose, object_pose_dict),
                               object_pose_dict, g), best_eval_total
