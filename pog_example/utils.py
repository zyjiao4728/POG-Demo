import logging
from typing import Tuple
from pog.algorithm.utils import *
from pog.algorithm.annealing import simulated_annealing
from pog.algorithm.structure import simulated_annealing_structure
from pog.algorithm.genetic import genetic_programming
from pog.graph.chromosome import to_graph, toChromosome
from pog.graph.graph import Graph
import time
import vedo

from pog.graph.shape import AffordanceType

__all__ = ['optimize', 'optimize_scene', 'optimize_structure']


def optimize(g=None,
             testname=None,
             scene_fn=None,
             edge_id=None,
             visualize=False,
             file_path=None,
             **kwargs):
    if g is None:
        try:
            g = Graph(testname, fn=scene_fn)
        except:
            logging.error('Unable to initialize scene graph.')
    if edge_id is not None: pose = g.getPose(edge_id=edge_id)
    else: pose = g.getPose()
    object_pose_dict, _ = gen_bound(pose)

    start = time.time()
    best, _ = simulated_annealing([objective], g, **kwargs)
    end = time.time()
    logging.info('Finished optimize in {:.4f} seconds'.format(end - start))

    g.setPose(arr2pose(best, object_pose_dict, pose))

    if file_path is not None:
        g.genMesh(file_path)

    if visualize:
        g.create_scene()
        vedo.show(g.scene.dump(concatenate=True), axes=1)

    logging.info(checkConstraints([], object_pose_dict, g))
    return g


def optimize_scene(g=None,
                   testname=None,
                   scene_fn=None,
                   visualize=False,
                   file_path=None,
                   **kwargs) -> Tuple[Graph, dict]:
    if g is None:
        try:
            g = Graph(testname, fn=scene_fn)
        except:
            logging.error('Unable to initialize scene graph.')

    start = time.time()
    if 'fixed_nodes' not in kwargs.keys():
        fixed_nodes = []
        for key, value in g.edge_dict.items():
            if value.relations[AffordanceType.Support]['dof'] == 'fixed':
                fixed_nodes.append(key)
        g, cnt_sat, best_eval_total = simulated_annealing_structure(
            g, fixed_nodes=fixed_nodes, **kwargs)
    else:
        g, cnt_sat, best_eval_total = simulated_annealing_structure(
            g, **kwargs)

    end = time.time()
    logging.info(
        'Finished optimize_scene in {:.4f} seconds. STAB_COST: {:.4f}; CNT_SAT: {}'
        .format(end - start, best_eval_total, cnt_sat))

    if file_path is not None:
        g.genMesh(file_path)

    if visualize:
        g.create_scene()
        vedo.show(g.scene.dump(concatenate=True), axes=1, screensize='auto')

    return g


def optimize_structure(g=None,
                       testname=None,
                       scene_fn=None,
                       visualize=False,
                       file_path=None,
                       **kwargs):
    if g is None:
        try:
            g = Graph(testname, fn=scene_fn)
        except:
            try:
                node_dict = kwargs['node_dict']
            except KeyError:
                logging.error('Unable to load node dict.')
    else:
        node_dict = g.node_dict

    start = time.time()
    best_chromo = genetic_programming(
        **kwargs) if 'node_dict_lite' in kwargs else genetic_programming(
            node_dict=to_node_lite(node_dict), **kwargs)
    end = time.time()
    logging.info(
        'Finished optimize_structure in {:.4f} seconds. Fitness: {:.4f}; CNT_SAT: {}'
        .format(end - start, best_chromo.fitness, best_chromo.cnt_sat))

    if g is not None:
        return best_chromo, optimize_scene(g=to_graph(best_chromo, node_dict,
                                                      g),
                                           visualize=visualize,
                                           file_path=file_path,
                                           fixed_nodes=list(best_chromo.fixed_nodes.keys()),
                                           **kwargs)
    else:
        return best_chromo, optimize_scene(g=to_graph(best_chromo, node_dict),
                                           visualize=visualize,
                                           file_path=file_path,
                                           fixed_nodes=list(best_chromo.fixed_nodes.keys()),
                                           **kwargs)
