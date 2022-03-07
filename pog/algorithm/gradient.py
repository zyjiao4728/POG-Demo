import numpy as np
from numpy.random import rand, randn

from pog.algorithm.utils import *
from pog.graph.graph import Graph

from pog.algorithm.params import STEP_SIZE, MAX_ITER, EPS_THRESH, TEMP_COEFF, NUM_CYCLES_EPS


# simulated annealing algorithm
def gradient_descent(
    objective,
    sg: Graph,
    node_id=None,
    random_start=False,
    verbose=False,
):

    if node_id is not None:
        pose = sg.getPose(edge_id=node_id)
    else:
        pose = sg.getPose()
    object_pose_dict, bounds = gen_bound(pose)

    # generate an initial point
    if random_start:
        best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
        sg.setPose(arr2pose(best, object_pose_dict, pose))
    else:
        best = pose2arr(pose, object_pose_dict, [])

    # evaluate the initial point
    best_eval, step_direction = objective[0](best, object_pose_dict, sg)
    # current working solution
    curr, curr_eval = best, best_eval

    # run the algorithm
    i = 0
    best_eval_arr = []
    history = []
    t = 1
    while i < MAX_ITER:
        i += 1
        # take a step
        gauss_noise = t * randn(len(curr))  # Gaussian
        t = t * TEMP_COEFF
        candidate = curr + (step_direction + gauss_noise) * STEP_SIZE

        # print(step_direction)

        # evaluate candidate point
        pose = arr2pose(candidate, object_pose_dict, pose)
        sg.setPose(pose)
        candidate_eval, temp_step_direction = objective[0](candidate,
                                                           object_pose_dict,
                                                           sg)

        # difference between candidate and current point evaluation
        diff = candidate_eval - curr_eval

        # calculate metropolis acceptance criterion

        # check for new best solution
        if candidate_eval < best_eval:
            # store new best point
            best, best_eval = candidate, candidate_eval
            # report progress
            if verbose:
                best_tmp = [float("{:.4f}".format(x)) for x in list(best)]
                print('>{} f({}) = {:.4f}'.format(i, best_tmp, best_eval))

            best_eval_arr.append(diff)
            if len(best_eval_arr) > NUM_CYCLES_EPS and (abs(
                    np.array(best_eval_arr[-NUM_CYCLES_EPS:])) <
                                                        EPS_THRESH).all():
                break

        if diff < 0:
            # store the new current point
            curr, curr_eval = candidate, candidate_eval
            step_direction = temp_step_direction

    return best, best_eval
