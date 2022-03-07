import time, logging, vedo, argparse
import numpy as np
from pog.graph.graph import Graph
from pog_example.utils import *
from pog.algorithm.structure import simulated_annealing_structure
from pog.algorithm.genetic import genetic_programming
from pog.algorithm.utils import to_node_lite
from pog.graph.chromosome import to_graph, Gene
from pog.graph import shape, node, edge
from pog.planning.planner import test, Searcher
from pog.planning.problem import PlanningOnGraphProblem
from pog.planning.utils import *

COLORS = {
    "light grey": [0.76470588, 0.76470588, 0.76470588, 1.],
    "red": [0.9254902, 0.10980392, 0.14117647, 1.],
    "blue": [0., 0.4470, 0.7410, 1.],
    "orange": [0.8500, 0.3250, 0.0980, 1.],
    "green": [0.4660, 0.6740, 0.1880, 1.],
    "yellow": [0.9290, 0.6940, 0.1250, 1.],
    "purple": [0.4940, 0.1840, 0.5560, 1.],
    "brown": [0.7255, 0.4784, 0.3373, 1.],
    "transparent": [0, 0, 0, 0],
    "light blue": [0.3010, 0.7450, 0.9330, 1.0],
    "dark red": [0.6350, 0.0780, 0.1840, 1.],
    "dark grey": [0.34509804, 0.34509804, 0.34509804, 1.],
}

COLOR_DICT = {
    0: "light grey",
    1: "red",
    2: "blue",
    3: "orange",
    4: "green",
    5: "yellow",
    6: "purple",
    7: "brown",
    8: "dark red",
    9: "light blue",
    10: "dark grey"
}


def create_exp1_scene(N_objects=1, b_r=0.2, d_r=0.02, h=0.02):
    node_dict = {}
    edge_dict = {}

    ground = node.Node(id=0, shape=shape.Box(size=[1.5, 1.5, 0.01]))
    node_dict[0] = ground
    ground.shape.shape.visual.face_colors[:] = np.array(
        COLORS[COLOR_DICT[0]]) * 255

    root_id = 0

    for i in range(1, N_objects + 1):
        temp_shape = shape.Cylinder(radius=b_r - (i - 1) * d_r, height=h)
        temp_shape.shape.visual.face_colors[:] = np.array(
            COLORS[COLOR_DICT[i % 10]]) * 255
        node_dict[i] = node.Node(id=i, shape=temp_shape)
        edge_dict[i] = edge.Edge(parent=root_id, child=i)
        edge_dict[i].add_relation(ground.affordance['box_aff_pz'],
                                  node_dict[i].affordance['cylinder_aff_nz'],
                                  dof_type='x-y',
                                  pose=[(np.random.rand() - 1 / 2) * 2.,
                                        (np.random.rand() - 1 / 2) * 2., 0.0])

    return node_dict, edge_dict, root_id


def createProblem(g: Graph):
    node_dict = g.node_dict
    node_dict_lite = to_node_lite(node_dict)

    constraints = {}
    fixed_nodes = {}
    for key, value in g.edge_dict.items():
        if value.relations[shape.AffordanceType.Support]['dof'] == 'fixed':
            fixed_nodes[key] = Gene(
                {
                    'node_id':
                    value.parent_id,
                    'name':
                    value.relations[shape.AffordanceType.Support]
                    ['parent'].name
                }, {
                    'node_id':
                    value.child_id,
                    'name':
                    value.relations[shape.AffordanceType.Support]['child'].name
                })

    fixed_nodes[g.root] = Gene({
        'node_id': g.root,
        'name': 'root'
    }, {
        'node_id': g.root,
        'name': 'root'
    })
    constraints['fixed'] = fixed_nodes
    constraints['propagation'] = {}

    for i in g.node_dict.keys():
        if i == g.root:
            continue
        else:
            constraints['propagation'][i] = i - 1

    return node_dict, node_dict_lite, constraints


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int, help='Number of disks between 1 and 10')
    parser.add_argument('-viewer',
                        action='store_true',
                        help='Enable the viewer and visualizes the plan')
    args = parser.parse_args()
    print('Arguments:', args)

    assert args.n >= 1 and args.n <= 10

    logFormatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)s] [%(levelname)-5.5s]  %(message)s"
    )
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(
        "pog_example/iros_2022_exp/exp1/test.log")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(logging.INFO)

    MAX_TRIES = 10
    # Initialize scene
    g_start = Graph('exp1', fn=create_exp1_scene, N_objects=args.n)

    # Initialize a random config
    cnt_sat = False
    count = 0
    while not cnt_sat and count < MAX_TRIES:
        count += 1
        g_start, cnt_sat, best_eval_total = simulated_annealing_structure(
            g_start, verbose=False, random_start=False)

    g_start.create_scene()

    node_dict, node_dict_lite, constraints = createProblem(g_start)
    best_chromo = genetic_programming(g=g_start,
                                      root=0,
                                      root_aff='box_aff_pz',
                                      node_dict=node_dict,
                                      node_dict_lite=node_dict_lite,
                                      constraints=constraints,
                                      method='heuristic',
                                      multiprocess=True)

    g_goal = optimize_scene(
        g=to_graph(best_chromo, node_dict),
        visualize=False,
        reverse=False,
        random_start=False,
        optim_method='gradient',
        visualize_step=False,
    )

    g_goal.create_scene()

    start = time.time()
    path = test(Searcher,
                problem=PlanningOnGraphProblem(g_start,
                                               g_goal,
                                               parking_place=0))
    end = time.time()
    print("Planning finished in time: {}".format(end - start))

    action_seq = path_to_action_sequence(path)

    _, succ = apply_action_sequence_to_graph(g_start,
                                             g_goal,
                                             action_seq,
                                             visualize=args.viewer,
                                             save_step=True)
