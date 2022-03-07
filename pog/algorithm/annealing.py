from copy import deepcopy
import logging
import numpy as np
from numpy.random import rand, randn

from pog.algorithm.utils import *
from pog.graph.graph import Graph

from pog.algorithm.params import STEP_SIZE, MAX_ITER, MAX_TEMP, TEMP_COEFF, \
 NUM_CYCLES_EPS, NUM_CYCLES_STEP, NUM_CYCLES_TEMP, EPS_THRESH, MAX_STEP_SIZE


# simulated annealing algorithm
def simulated_annealing(
    objective,
    sg: Graph,
    node_id=None,
    random_start=False,
    verbose=False,
    method='standard',
):
    """simulated annealing algorithm to maximum stability.

		NOTE: This function is only for scene graph with max depth = 2

	Args:
		objective (list): a list of objective functions
		sg (Graph): scene graph
		node_id (list, optional): a list of nodes to be optimized. Defaults to None.
		random_start (bool, optional): Randomly select initial configuration. Defaults to False.
		verbose (bool, optional): More outputs. Defaults to False.

	Returns:
		best: best configuration
		best_eval: cost of best configuration
	"""
    if method == 'adaptive':
        return adaptive_simulated_annealing(objective,
                                            sg,
                                            node_id,
                                            random_start=random_start,
                                            verbose=verbose)

    if node_id is not None:
        pose = sg.getPose(edge_id=node_id)
    else:
        pose = sg.getPose()
    object_pose_dict, bounds = gen_bound(pose)

    # generate an initial point
    if random_start:
        best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    else:
        best = pose2arr(pose, object_pose_dict, [])

    # evaluate the initial point
    best_eval, _ = objective[0](best, object_pose_dict, sg)
    # current working solution
    curr, curr_eval = best, best_eval

    # run the algorithm
    t = MAX_TEMP
    best_eval_arr = []
    history = []
    for i in range(MAX_ITER):
        # take a step
        step_direction = randn(len(curr))  # Gaussian
        # step_direction = 2. * (rand(len(curr)) - 0.5) # Uniform
        candidate = curr + step_direction / np.linalg.norm(
            step_direction) * STEP_SIZE

        # evaluate candidate point
        pose = arr2pose(candidate, object_pose_dict, pose)
        sg.setPose(pose)
        candidate_eval, _ = objective[0](candidate, object_pose_dict, sg)
        # difference between candidate and current point evaluation
        diff = candidate_eval - curr_eval

        # calculate temperature for current epoch
        t = t * TEMP_COEFF

        # calculate metropolis acceptance criterion
        metropolis = np.exp(min(-diff / t, 700.))  # <- avoid overflow

        # check for new best solution
        if candidate_eval < best_eval:
            # store new best point
            best, best_eval = candidate, candidate_eval
            # report progress
            if verbose:
                best_tmp = [float("{:.4f}".format(x)) for x in list(best)]
                logging.debug('>{} f({}) = {:.4f}, temp: {:.4f}'.format(
                    i, best_tmp, best_eval, t))

            best_eval_arr.append(diff)
            if len(best_eval_arr) > NUM_CYCLES_EPS and (abs(
                    np.array(best_eval_arr[-NUM_CYCLES_EPS:])) <
                                                        EPS_THRESH).all():
                break

        # check if we should keep the new point
        if diff < 0 or rand() < metropolis:
            # store the new current point
            curr, curr_eval = candidate, candidate_eval

    return best, best_eval


def adaptive_simulated_annealing(objective,
                                 sg: Graph,
                                 node_id=None,
                                 random_start=False,
                                 verbose=False):

    if node_id is not None:
        pose = sg.getPose(edge_id=node_id)
    else:
        pose = sg.getPose()
    object_pose_dict, bounds = gen_bound(pose)

    # generate an initial point
    if random_start:
        best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    else:
        best = pose2arr(pose, object_pose_dict, [])

    # evaluate the initial point
    best_eval = objective[0](best, object_pose_dict, sg)
    # current working solution
    curr, curr_eval = best, best_eval

    eval_arr, counts_cycles, counts_resets = [], 0, 0
    n = len(curr)
    a = np.zeros(n)
    step_vector = MAX_STEP_SIZE * np.ones(n)
    c = 0.1 * np.ones(n)
    t = MAX_TEMP
    # run the algorithm
    i = 0
    while i < 5000:
        i += 1
        for iter in range(n):
            step = np.zeros(n)
            step[iter] = 2 * (rand() - 0.5) * step_vector[iter]
            temp = curr + step
            temp_eval = objective[0](temp, object_pose_dict, sg)
            diff_temp_eval = temp_eval - curr_eval
            if diff_temp_eval < 0 or rand() < np.exp(-diff_temp_eval / t):
                curr, curr_eval = temp, temp_eval
                a[iter] += 1.
                if curr_eval < best_eval:
                    best, best_eval = curr, curr_eval

        counts_cycles += 1
        if counts_cycles <= NUM_CYCLES_STEP: continue

        counts_cycles = 0
        step_vector = corana_update(step_vector, a, c, NUM_CYCLES_STEP)
        a = np.zeros(n)
        counts_resets += 1
        if counts_resets <= NUM_CYCLES_TEMP: continue

        t *= TEMP_COEFF
        counts_resets = 0
        eval_arr.append(curr_eval)
        if not (len(eval_arr) > NUM_CYCLES_EPS and eval_arr[-1] - best_eval <= EPS_THRESH and \
         (abs((eval_arr[-1] - np.array(eval_arr))[-NUM_CYCLES_EPS:])<= EPS_THRESH).all()):
            curr, curr_eval = best, best_eval
            if verbose:
                best_tmp = [float("{:.4f}".format(x)) for x in list(best)]
                print('>{} f({}) = {:.4f}, temp: {:.4f}'.format(
                    i, best_tmp, best_eval, t))

        else:
            break

    return [best, best_eval]


def corana_update(v, a, c, ns):
    for i in range(len(v)):
        ai, ci = a[i], c[i]

        if ai > 0.6 * ns:
            v[i] *= (1 + ci * (ai / ns - 0.6) / 0.4)
        elif ai < 0.4 * ns:
            v[i] /= (1 + ci * (0.4 - ai / ns) / 0.4)

    return v
