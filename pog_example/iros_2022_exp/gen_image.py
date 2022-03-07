import json
import logging
from time import sleep
from pog.algorithm.genetic import genetic_programming
from pog.algorithm.utils import objective_history, pose2arr, to_node_lite, gen_bound
from pog.graph.chromosome import Chromosome, Gene, to_graph, toChromosome
from pog.graph.edge import Edge
from pog.graph.environment import Environment
from pog.graph.graph import Graph
from pog.graph.node import Node
from pog.graph import shape
from pog.graph.shapes import OpenShelf
from pog.graph.shapes.cone import Cone
from pog_example.test_main.init_logger import rootLogger
import vedo

from pog_example.utils import optimize_scene, optimize_structure


def create_test_scene():
    node_ground = Node(id=0, shape=shape.Box(size=[2., 2., 0.01]))
    node_shelf = Node(id=1, shape=OpenShelf(size=[0.45, 0.35, 0.25]))
    node_plate = Node(id=2, shape=shape.Cylinder(radius=0.2, height=0.04))
    node_box = Node(id=3, shape=shape.Box(size=[0.16, 0.22, 0.3]))
    node_cone1 = Node(id=4, shape=Cone(height=0.45, radius=0.08))
    node_cone2 = Node(id=5, shape=Cone(height=0.25, radius=0.12))

    node_dict = init_node_dict(node_ground, node_shelf, node_plate, node_box,
                               node_cone1, node_cone2)

    edge_ground_shelf = Edge(parent=0, child=1)
    edge_ground_shelf.add_relation(node_ground.affordance['box_aff_pz'],
                                   node_shelf.affordance['shelf_outer_bottom'],
                                   dof_type='fixed',
                                   pose=[0., 0., 0.])

    edge_shelf_plate = Edge(parent=1, child=2)
    edge_shelf_plate.add_relation(
        node_shelf.affordance['shelf_outer_top'],
        node_plate.affordance['cylinder_aff_nz'],
        # dof_type='x-y',
    )
    edge_shelf_box = Edge(1, 3)
    edge_shelf_box.add_relation(node_shelf.affordance['shelf_outer_top'],
                                node_box.affordance['box_aff_nz'])

    edge_shelf_cone1 = Edge(1, 4)
    edge_shelf_cone1.add_relation(node_shelf.affordance['shelf_outer_top'],
                                  node_cone1.default_affordance)

    edge_shelf_cone2 = Edge(1, 5)
    edge_shelf_cone2.add_relation(node_shelf.affordance['shelf_outer_top'],
                                  node_cone2.default_affordance)

    edge_dict = init_edge_dict(edge_ground_shelf, edge_shelf_plate,
                               edge_shelf_box, edge_shelf_cone1,
                               edge_shelf_cone2)
    root_id = 0

    return node_dict, edge_dict, root_id


def create_demo_scene():
    node_ground = Node(id=0, shape=shape.Box(size=[2., 2., 0.01]))
    # node_shelf = Node(id=1, shape=OpenShelf(size=[0.45, 0.35, 0.25]))
    node_plate = Node(id=2, shape=shape.Cylinder(radius=0.2, height=0.03))
    node_box = Node(id=3, shape=shape.Box(size=[0.15, 0.21, 0.3]))
    node_cone1 = Node(id=4, shape=Cone(height=0.25, radius=0.12))
    node_cone2 = Node(id=5, shape=Cone(height=0.4, radius=0.08))

    node_dict = init_node_dict(node_ground, node_plate, node_box, node_cone1,
                               node_cone2)

    edge_ground_plate = Edge(parent=0, child=2)
    edge_ground_plate.add_relation(node_ground.affordance['box_aff_pz'],
                                   node_plate.affordance['cylinder_aff_nz'],
                                   dof_type='fixed',
                                   pose=[-0.25, 0.25, 0.])

    # edge_ground_box = Edge(0, 3)
    # edge_ground_box.add_relation(node_ground.affordance['box_aff_pz'],
    #                              node_box.affordance['box_aff_nz'],
    #                              pose=[0.2, 0.2, 0.])

    # edge_ground_cone1 = Edge(0, 4)
    # edge_ground_cone1.add_relation(node_ground.affordance['box_aff_pz'],
    #                                node_cone1.default_affordance,
    #                                pose=[0.6, 0.6, 0.])

    # edge_ground_cone2 = Edge(0, 5)
    # edge_ground_cone2.add_relation(node_ground.affordance['box_aff_pz'],
    #                                node_cone2.default_affordance,
    #                                pose=[-0.6, -0.6, 0.])

    edge_dict = init_edge_dict(
        edge_ground_plate,
        # edge_ground_box,
        # edge_ground_cone1,
        # edge_ground_cone2,
    )
    root_id = 0

    return node_dict, edge_dict, root_id


def init_node_dict(*nodes):
    node_dict = {}
    for node in nodes:
        node_dict[node.id] = node
    return node_dict


def init_edge_dict(*edges):
    edge_dict = {}
    for edge in edges:
        edge_dict[edge.child_id] = edge
    return edge_dict


def create_problem(g: Graph):
    node_dict = g.node_dict
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
    constraints['propagation'] = {2: 1, 3: 1, 4: 1, 5: 1}
    constraints['nocrossoverto'] = [g.root]
    constraints['height'] = 0.7
    return node_dict, to_node_lite(node_dict), constraints


def build_history_graph(chromosome: dict, node_dict: dict, root: int):

    def fn():
        edge_dict = {}

        for item in chromosome.values():
            edge_dict[item.child_id] = Edge(parent=item.parent_id,
                                            child=item.child_id)
            edge_dict[item.child_id].add_relation(node_dict[item.parent_id].affordance[item.parent_affordance_name],\
                    node_dict[item.child_id].affordance[item.child_affordance_name], pose=[0.,0.,0.])

        root_id = root
        return node_dict, edge_dict, root_id

    return Graph('Graph of history', fn=fn)


if __name__ == '__main__':
    g = Graph('test graph', fn=create_test_scene)
    sg = g.getSubGraph(0)
    sg.create_scene()
    init_chromo = toChromosome(g).chromosome
    # vedo.show(sg.scene.dump(concatenate=True), axes=1)
    # g.show()
    # Environment(sg)
    # exit(0)
    # g.show()
    node_dict, node_dict_lite, constraints = create_problem(g)
    best_chromo, final_g = optimize_structure(g,
                                              'test',
                                              visualize=False,
                                              node_dict_lite=node_dict_lite,
                                              root=g.root,
                                              constraints=constraints,
                                              initchromo=init_chromo)
    print(best_chromo.history_str)

    g.create_scene()
    Environment(g, display=False, input_required=False)
    for i, chromo in enumerate(best_chromo.history):
        g_next = build_history_graph(chromo, node_dict, g.root)
        # g_next_optimized = optimize_scene(g_next, 'test')
        g_next_optimized = g_next
        g_next_optimized.create_scene()
        # g_next_optimized.scene.show()
        Environment(g_next_optimized,
                    display=False,
                    export_image_path='./result/step{}.png'.format(i),
                    input_required=False)

    g_final = g_next_optimized
    g_final.create_scene()
    # g_final.scene.show()

    g_final_optimized, pose_history = optimize_scene(
        g_final.copy(),
        visualize_step=False,
        fixed_nodes=[1],
    )
    loss_dict = {}
    for node_id, optimization_history in pose_history.items():
        total_steps = len(optimization_history)
        loss_dict[node_id] = []
        logging.info('total steps for node_id {}: {}'.format(node_id, total_steps))
        for j, pose in enumerate(optimization_history):
            object_pose_dict,_ = gen_bound(g_final.getPose())
            current_loss, _ = objective_history(pose2arr(g_final.getPose(), object_pose_dict), object_pose_dict, g_final)
            logging.info('current loss: {}'.format(current_loss))
            loss_dict[node_id].append(current_loss)
            g_final.setPose(pose)
            g_final.create_scene()
            if j % (total_steps / 4) != 0:
                pass
            else:
                # g_final.scene.show()
                Environment(g_final, display=True, input_required=True)
    with open('./result/loss_dump.json', 'w') as f:
        json.dump(loss_dict, f)
    # print(best_chromo.history)
    Environment(final_g, input_required=True)
