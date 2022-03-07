from typing import List

from pog.graph.graph import Graph
from pog.planning.action import Action, ActionType

import copy

class SearchNode():
    def __init__(self, action_seq : List, constraints : List = [], action : Action = None, current : Graph = None, goal : Graph = None) -> None:
        self.action_sequence = action_seq
        self.constraints = constraints
        self.unexplored = list(range(len(self.action_sequence)))
        self.unexplored.reverse()
        self.action = action
        if action is not None and self.action.action_type == ActionType.Pick:
            self.occupied = True
        else:
            self.occupied = False
        self.cost = 1 # unit cost for now
        self.is_root = True if self.action is None else False
        self.current = current
        self.goal = goal
    
    def __eq__(self, other) -> bool:
        return (self.action_sequence == other.action_sequence) and (self.action == other.action)
    
    def __repr__(self) -> str:
        if self.action is None:
            return str("Start")
        else:
            return str(self.action)

    def selectAction(self):
        while self.unexplored:
            explore_idx = self.unexplored.pop()
            if self.invalidSeletion(self.action_sequence[explore_idx]) : continue
            new_action_seq = copy.deepcopy(self.action_sequence)
            new_constraints = copy.deepcopy(self.constraints)
            new_action = new_action_seq.pop(explore_idx)
            yield (new_action_seq, new_constraints, new_action)
    
    def invalidSeletion(self, action):
        return (self.constraints.violate(action)) or (self.occupied and action.action_type != ActionType.Place)
            
    def updateCost(self):
        pass

    @property
    def is_leaf_node(self):
        return False if self.action_sequence else True
