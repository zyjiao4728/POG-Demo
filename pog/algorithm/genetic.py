import networkx as nx
from numpy.random import rand
import multiprocessing as mp
import random

from pog.graph.chromosome import Chromosome
from pog.algorithm.utils import *
from pog.algorithm.structure import simulated_annealing_structure

import time, copy, logging

from pog.algorithm.params import MUTATION_PROBABILITY, CROSSOVER_CHILDREN, \
    MAX_POPULATION, MAX_GENERATION, AREA_RATIO_MARGIN, FITNESS_THRESH, TOURNAMENT_K


def computeAreaHeuristic(chromo):
    """Compute area heuristic

    Args:
        chromo (Chromosome): Chromosome to compute heuristic

    Returns:
        satisfy (bool): True if all area constraints are satisfied
        cost (int): Accumulated violation cost
    """
    chromo.trackDepth()
    max_depth = len(nx.algorithms.dag_longest_path(chromo.chromograph))
    cost = 0
    satisfy = True
    for depth in range(max_depth):
        for node in chromo.depth_dict[depth]:
            ratio = {}
            n_succ = {}
            for succ in chromo.chromograph.successors(node):

                if chromo.chromosome[
                        succ].parent_affordance_name not in n_succ.keys():
                    n_succ[chromo.chromosome[succ].parent_affordance_name] = 1
                else:
                    n_succ[chromo.chromosome[succ].parent_affordance_name] += 1

                area_2 = chromo.node_dict[succ].affordance[
                    chromo.chromosome[succ].child_affordance_name]['area']
                area_1 = chromo.node_dict[node].affordance[
                    chromo.chromosome[succ].parent_affordance_name]['area']

                if chromo.chromosome[
                        succ].parent_affordance_name not in ratio.keys():
                    ratio[chromo.chromosome[succ].
                          parent_affordance_name] = area_2 / area_1
                else:
                    ratio[chromo.chromosome[succ].
                          parent_affordance_name] += area_2 / area_1

            for key, value in ratio.items():
                if value > AREA_RATIO_MARGIN and n_succ[key] > 1:
                    satisfy = False

                cost += max(value, AREA_RATIO_MARGIN) - AREA_RATIO_MARGIN

    return satisfy, cost


def computeFitness(chromo: Chromosome, method='heuristic', **kwargs):
    """Compute fitness of chromosome, the lower, the better.

    Args:
        chromo (Chromosome): Chromosome to compute fitness
        method (str): Method of computing heuristics

    Returns:
        chromo.cnt_sat (bool): True if all constraints are satisfied
        chromo.fitness (int): Fitness of current chromosome
    """
    if method == 'structure':  # Extremly time-consuming
        _, stable_cnt_sat, stable_cost = simulated_annealing_structure(
            chromo.to_graph(), fixed_nodes=chromo.fixed_nodes, **kwargs)
        prop_cnt_sat, prop_cost = chromo.checkPropConstraints()
        height_cnt_sat, height_cost = chromo.checkHeightConstraints()
        chromo.fitness = stable_cost + prop_cost + height_cost
        chromo.cnt_sat = stable_cnt_sat and prop_cnt_sat and height_cnt_sat
    elif method == 'heuristic':
        cnt_sat, cost = computeAreaHeuristic(chromo)
        prop_cnt_sat, prop_cost = chromo.checkPropConstraints()
        height_cnt_sat, height_cost = chromo.checkHeightConstraints()
        cont_cnt_sat, cont_cost = chromo.checkContConstraints()
        chromo.fitness = cost + prop_cost + height_cost + cont_cost
        chromo.cnt_sat = cnt_sat and prop_cnt_sat and height_cnt_sat and cont_cnt_sat
    else:
        logging.error(
            'Unsupported method {}. Supported method: structure or heuristic.'.
            format(method))
    return chromo.cnt_sat, chromo.fitness


def tournament_selection(population):
    selected_population = []
    for i in range(MAX_POPULATION):
        candidates = random.sample(population, TOURNAMENT_K)
        candidates.sort()
        selected_population.append(candidates[0])
    return selected_population


def genetic_programming(**kwargs):
    """genetic programming for structure search

        NOTE: Multiprocessing only work for 'heuritic' method.

    Returns:
        (chromosome): Best fit chromosome
    """
    population = []
    method = kwargs['method'] if 'method' in kwargs.keys() else 'heuristic'
    multiprocess = kwargs['multiprocess'] if 'multiprocess' in kwargs.keys(
    ) else False

    for idx in range(MAX_POPULATION):
        population.append(Chromosome(**kwargs))
        if population[-1].initialize():
            computeFitness(population[-1], method)
        else:
            population.pop()

    population.sort()

    if multiprocess:
        logging.info('Start multi-processing.')
    else:
        logging.info('Start single-processing.')

    for n_gen in range(MAX_GENERATION):
        start = time.time()
        logging.debug('Generation: {}'.format(n_gen + 1))
        population = tournament_selection(population)
        temp_population = []
        if multiprocess:
            with mp.Pool(processes=mp.cpu_count()) as pool:
                temp_population = pool.map(genetic_programming_helper, [
                    population[i * mp.cpu_count():(i + 1) * mp.cpu_count()]
                    for i in range((len(population) + mp.cpu_count() - 1) //
                                   mp.cpu_count())
                ])
            population = temp_population[0]
        else:
            for chromo in population:
                for i in range(CROSSOVER_CHILDREN):
                    new_chromo = copy.deepcopy(chromo)
                    new_chromo.crossover()
                    if rand() < MUTATION_PROBABILITY:
                        new_chromo.mutate()
                    _, _ = computeFitness(new_chromo, method)
                    temp_population.append(new_chromo)
            population = temp_population
        population.sort()
        population_sat = [ch for ch in population if ch.cnt_sat == True]
        end = time.time()
        if population_sat and population_sat[0].fitness < FITNESS_THRESH:
            logging.info(
                'Finished generation {} in {:.4f} seconds. Solution Found! Fitness: {:.4f}.'
                .format(n_gen + 1, end - start, population_sat[0].fitness))
            return population_sat[0]
        else:
            logging.info(
                'Finished generation {} in {:.4f} seconds. Best fitness: {:.4f}.'
                .format(n_gen + 1, end - start, population[0].fitness))
    return population[0]


def genetic_programming_helper(population, method='heuristic'):
    temp_population = []
    for chromo in population:
        for i in range(CROSSOVER_CHILDREN):
            new_chromo = copy.deepcopy(chromo)
            new_chromo.crossover()
            if rand() > MUTATION_PROBABILITY:
                new_chromo.mutate()
            computeFitness(new_chromo, method)
            temp_population.append(new_chromo)
    return temp_population
