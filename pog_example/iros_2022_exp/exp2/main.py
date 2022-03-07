import logging, argparse
from pog.graph.graph import Graph
from pog.planning.planner import test, Searcher
from pog.planning.problem import PlanningOnGraphProblem
from pog.planning.utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-viewer',
                        action='store_true',
                        help='Enable the viewer and visualizes the plan')
    args = parser.parse_args()
    print('Arguments:', args)

    logFormatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)s] [%(levelname)-5.5s]  %(message)s"
    )
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler("pog_example/iros_2022_exp/exp2/test.log")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(logging.INFO)

    # Planning
    g_start = Graph('exp2-init', file_dir='pog_example/iros_2022_exp/exp2/', file_name='init.json')
    g_goal = Graph('exp2-goal', file_dir='pog_example/iros_2022_exp/exp2/', file_name='goal.json')
    
    # Environment(g_goal)
    path = test(Searcher, problem=PlanningOnGraphProblem(g_start, g_goal, parking_place=99))
    
    action_seq = path_to_action_sequence(path)            
    apply_action_sequence_to_graph(g_start, g_goal, action_seq, visualize=args.viewer)
