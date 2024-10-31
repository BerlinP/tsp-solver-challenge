"""Solver implementations for TSP problems"""
from typing import List, Union
import numpy as np
from abc import ABC, abstractmethod
import random
import asyncio
import concurrent.futures
import time
from evaluation_utils import GraphV2Problem
from graph_utils import valid_problem

DEFAULT_SOLVER_TIMEOUT = 30

        
class BaseSolver(ABC):
    """Base class for all solvers"""
    def __init__(self, problem_types:List[Union[GraphV2Problem]]):
        self.problem_types = [problem.problem_type for problem in problem_types] # defining what problems the solver is equipped to solve
        self.future_tracker = {}

    @abstractmethod
    async def solve(self, formatted_problem, future_id, *args, **kwargs)->List[int]:
        '''
        Abstract class that handles the solving of GraphV1Problems contained within the Synapse.

        Solvers can be developed to handle multiple types of graph traversal problems.

        It takes a formatted_problem (post-transformation) as an input and returns the optimal path based on the objective function (optional)
        '''
        ...
    
    @abstractmethod
    def problem_transformations(self, problem: GraphV2Problem):
        '''
        This abstract class applies any necessary transformation to the problem to convert it to the form required for the solve method
        '''
        ...

    def is_valid_problem(self, problem):
        '''
        checks if the solver is supposed to be able to solve the given problem and that the problem specification is valid.
        Note that this does not guarantee the problem has a solution. For example: the TSP problem might be a partially connected graph with no hamiltonian cycle
        '''
        return valid_problem(problem) and problem.problem_type in self.problem_types

    async def solve_problem(self, problem: GraphV2Problem, timeout:int=DEFAULT_SOLVER_TIMEOUT):
        '''
        This method implements the security checks
        Then it makes the necessary transformations to the problem
        and passes it on to the solve method

        Checks for the integrity of the data (that the problem is legitimate) are handled outside the forward function
        '''
        if self.is_valid_problem(problem):

            future_id = id(problem)
            self.future_tracker[future_id] = False

            transformed_problem = self.problem_transformations(problem)
            
            loop = asyncio.get_running_loop()
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit the asynchronous task to the executor
                future = loop.run_in_executor(executor, lambda: asyncio.run(self.solve(transformed_problem,future_id)))
                try:
                    result = await asyncio.wait_for(future, timeout)
                    return result
                except asyncio.TimeoutError:
                    print(f"Task {future_id} timed out after: {time.time() - start_time}, with timeout set to {timeout}")
                    self.future_tracker[future_id] = True
                    return False
                except Exception as exc:
                    print(f"Task generated an exception: {exc}")
                    return False
        else:
            print(f"current solver: {self.__class__.__name__} cannot handle received problem: {problem.problem_type}")
            return False

class NearestNeighbourSolver(BaseSolver):
    """Nearest neighbor heuristic solver"""
    def __init__(self, problem_types:List[GraphV2Problem]=[GraphV2Problem()]):
        super().__init__(problem_types=problem_types)

    def problem_transformations(self, problem: GraphV2Problem):
        """Transform problem for nearest neighbor solver"""
        return problem.edges

    async def solve(self, formatted_problem, future_id: int) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix[0])
        visited = [False] * n
        route = []

        # Start from node 0
        current_node = 0
        route.append(current_node)
        visited[current_node] = True

        for _ in range(n - 1):
            if self.future_tracker.get(future_id):
                return None
                
            # Find nearest unvisited neighbor
            nearest_distance = np.inf
            nearest_node = random.choice([i for i, is_visited in enumerate(visited) if not is_visited])
            
            for j in range(n):
                if not visited[j] and distance_matrix[current_node][j] < nearest_distance:
                    nearest_distance = distance_matrix[current_node][j]
                    nearest_node = j

            route.append(nearest_node)
            visited[nearest_node] = True
            current_node = nearest_node
        
        # Return to start
        route.append(route[0])
        return route
