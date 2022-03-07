# Some functions are modified from searchProblem.py - representations of search problems http://aipython.org
import logging, heapq

from pog.planning.problem import PlanningOnGraphPath, PlanningOnGraphProblem
from pog.planning.utils import *


class Searcher():
    """returns a searcher for a problem.
    Paths can be found by repeatedly calling search().
    This does depth-first search unless overridden
    """

    def __init__(self, problem):
        """creates a searcher from a problem
        """
        self.problem = problem
        self.initialize_frontier()
        self.num_expanded = 0
        self.add_to_frontier(PlanningOnGraphPath(problem.start_node()))

    def initialize_frontier(self):
        self.frontier = []

    def empty_frontier(self):
        return self.frontier == []

    def add_to_frontier(self, path):
        self.frontier.append(path)

    def search(self):
        """returns (next) path from the problem's start node
        to a goal node. 
        Returns None if no path exists.
        """
        while not self.empty_frontier():
            path = self.frontier.pop()
            logging.debug("Expanding: {} (cost: {})".format(path, path.cost))
            self.num_expanded += 1
            if self.problem.is_goal(path.end()):  # solution found
                logging.info(
                    "{} paths have been expanded and {} paths remain in the frontier."
                    .format(self.num_expanded, len(self.frontier)))
                self.solution = path  # store the solution found
                return path
            else:
                neighs = self.problem.neighbors(path.end())
                logging.debug("Expanded {} neighbors.".format(len(neighs)))
                for arc in reversed(list(neighs)):
                    self.add_to_frontier(PlanningOnGraphPath(path, arc))
        logging.info("No (more) solutions. Total of {} paths expanded.".format(
            self.num_expanded))


class FrontierPQ():
    """A frontier consists of a priority queue (heap), frontierpq, of
        (value, index, path) triples, where
    * value is the value we want to minimize (e.g., path cost + h).
    * index is a unique index for each element
    * path is the path on the queue
    Note that the priority queue always returns the smallest element.
    """

    def __init__(self):
        """constructs the frontier, initially an empty priority queue 
        """
        self.frontier_index = 0  # the number of items ever added to the frontier
        self.frontierpq = []  # the frontier priority queue

    def empty(self):
        """is True if the priority queue is empty"""
        return self.frontierpq == []

    def add(self, path, value):
        """add a path to the priority queue
        value is the value to be minimized"""
        self.frontier_index += 1  # get a new unique index
        heapq.heappush(self.frontierpq, (value, -self.frontier_index, path))

    def pop(self):
        """returns and removes the path of the frontier with minimum value.
        """
        (_, _, path) = heapq.heappop(self.frontierpq)
        return path

    def count(self, val):
        """returns the number of elements of the frontier with value=val"""
        return sum(1 for e in self.frontierpq if e[0] == val)

    def __repr__(self):
        """string representation of the frontier"""
        return str([(n, c, str(p)) for (n, c, p) in self.frontierpq])

    def __len__(self):
        """length of the frontier"""
        return len(self.frontierpq)

    def __iter__(self):
        """iterate through the paths in the frontier"""
        for (_, _, path) in self.frontierpq:
            yield path


class AStarSearcher(Searcher):
    """returns a searcher for a problem.
    Paths can be found by repeatedly calling search().
    """

    def __init__(self, problem):
        super().__init__(problem)

    def initialize_frontier(self):
        self.frontier = FrontierPQ()

    def empty_frontier(self):
        return self.frontier.empty()

    def add_to_frontier(self, path):
        """add path to the frontier with the appropriate cost"""
        value = path.cost + self.problem.heuristic(path.end())
        self.frontier.add(path, value)


def test(SearchClass, problem: PlanningOnGraphProblem):
    """Unit test for aipython searching algorithms.
    SearchClass is a class that takes a problemm and implements search()
    problem is a search problem
    solutions is a list of optimal solutions 
    """
    logging.info("test: Testing problem 1:")
    schr1 = SearchClass(problem)
    path1 = schr1.search()
    logging.info("test: Path found: {}".format(path1))
    assert path1 is not None, "No path is found in problem1"
    logging.info("test: Passed unit test")
    return path1
