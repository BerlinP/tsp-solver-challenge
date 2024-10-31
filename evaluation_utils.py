"""Utility functions and classes for solver evaluation"""
from typing import List, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
import ast
import math
from graph_utils import (
    geom_edges,
    euc_2d_edges,
    man_2d_edges,
    is_valid_path
)
from protocol import (
    GraphV2Problem,
    GraphV2Synapse
)


def get_tour_distance(synapse:GraphV2Synapse)->float:
    '''
    Returns the total tour distance for the TSP problem as a float.

    Takes a synapse as its only argument
    '''
    
    problem = synapse.problem
    if 'TSP' not in problem.problem_type:
        raise ValueError(f"get_tour_distance is an invalid function for processing {problem.problem_type}")
    
    if not synapse.solution:
        return np.inf
    distance=np.nan
    if problem.directed or isinstance(synapse.problem, GraphV2Problem):
        # This is a General TSP problem
        # check if path and edges are of the appropriate size
        edges=problem.edges
        path=synapse.solution
        if isinstance(path,list):
            assert is_valid_path(path), ValueError('Provided path is invalid')
            assert len(path) == problem.n_nodes+1, ValueError('An invalid number of cities are contained within the provided path')

            distance = 0
            for i, source in enumerate(path[:-1]):
                destination = path[i+1]
                distance += edges[source][destination]
    else:
        # This is a metric TSP problem
        # check if path and coordinates are of the appropriate size
        coordinates=problem.nodes
        path=synapse.solution
        
        if isinstance(path,list):
            assert is_valid_path(path), ValueError('Provided path is invalid')
            assert len(path) == problem.n_nodes+1, ValueError('An invalid number of cities are contained within the provided path')

            # sort cities into pairs
            pairs = [(path[i], path[i+1]) for i in range(len(path)-1)]
            distance = 0
            for pair in pairs:
                distance += math.hypot(coordinates[pair[0]][0] - coordinates[pair[1]][0], coordinates[pair[0]][1] - coordinates[pair[1]][1])
    return distance if not np.isnan(distance) else np.inf

class MetricTSPV2Generator:
    """Generator for TSP problems"""
    @staticmethod
    def _problem_size(problem: GraphV2Problem) -> int:
        return problem.n_nodes // 1000 * 1000

    @staticmethod
    def recreate_edges(problem: GraphV2Problem, loaded_datasets: dict):
        """Recreate edge distances for a problem"""
        node_coords_np = loaded_datasets[problem.dataset_ref]["data"]
        node_coords = np.array([node_coords_np[i][1:] for i in problem.selected_ids])
        problem.nodes = node_coords
        
        if problem.cost_function == "Geom":
            problem.edges = geom_edges(node_coords).tolist()
        elif problem.cost_function == "Euclidean2D":
            problem.edges = euc_2d_edges(node_coords).tolist()
        elif problem.cost_function == "Manhatten2D":
            problem.edges = man_2d_edges(node_coords).tolist()
        else:
            return "Only Geom, Euclidean2D, and Manhatten2D supported for now."

    @staticmethod
    def generate_problems_from_existing_results(
        top_n: int,
        results_path: str,
        n_problems: int
    ) -> Tuple[List[GraphV2Problem], List[int], np.ndarray, List[float], List[float]]:
        """Generate problems from existing results file"""
        df = pd.read_csv(results_path, sep='\t')
        problems = []
        distances = []
        medians = []
        worst = []
        count = 0
        
        for _, row in df.iterrows():
            if count >= n_problems:
                break
                
            problem_type = row['problem_type']
            if problem_type == 'Metric mTSP':
                continue
                
            n_nodes = row['n_nodes']
            dataset_ref = row['dataset_ref']
            selected_node_idxs = ast.literal_eval(row['selected_ids'])
            distances_str = row['distances'].replace('inf', 'float("inf")')
            distances_arr = eval(distances_str)

            # Filter out infinite values and get top N minimum distances
            valid_distances = [d for d in distances_arr if d != float('inf')]
            top_n_distances = sorted(valid_distances)[:top_n]
            
            test_problem = GraphV2Problem(
                problem_type="Metric TSP",
                n_nodes=n_nodes,
                selected_ids=selected_node_idxs,
                cost_function="Geom",
                dataset_ref=dataset_ref
            )
            
            problems.append(test_problem)
            distances.append(top_n_distances)
            medians.append(np.median(valid_distances))
            worst.append(sorted(valid_distances)[-1])
            count += 1
            
        return problems, [MetricTSPV2Generator._problem_size(p) for p in problems], distances, medians, worst
