"""Evaluate TSP solvers using existing results"""
from typing import List
import pandas as pd
import tqdm
import time
import asyncio
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

from evaluation_utils import (
    GraphV2Problem,
    GraphV2Synapse,
    MetricTSPV2Generator,
    get_tour_distance,
)

from solvers import (
    NearestNeighbourSolver,
    OrtoolSolver
)

from dataset_utils import load_default_dataset

ROOT_DIR = "."
SAVE_DIR = "evaluation_results"
EXISTING_RESULTS_PATH = "latest_top_n.tsv"
TOP_N = 1
N_PROBLEMS = int(os.getenv('N_PROBLEMS')) if 'N_PROBLEMS' in os.environ else 5

def can_show_plot():
    """Check if plotting is possible in current environment"""
    if os.name == 'posix':
        display = os.getenv('DISPLAY')
        if not display:
            return False

    backend = matplotlib.get_backend()
    if backend in ['agg', 'cairo', 'svg', 'pdf', 'ps']:
        return False

    return True

def compute_relative_scores(scores_df: pd.DataFrame, tolerance=1e-5):
    """Compute relative scores between different solvers"""
    def normalize_row(row):
        min_val = row.min()
        max_val = row.max()
        if max_val > min_val:  # To avoid division by zero
            return (row - min_val) / (max_val - min_val)
        else:
            return row  # If all values are the same, return the original row
        
    relative_scores = pd.DataFrame(index=scores_df.index)
    solvers = scores_df.columns.difference(['problem_size'])

    for solver in solvers:
        relative_scores[solver] = scores_df[solvers].apply(
            lambda row: sum(
                np.isclose(row[solver], row[other_solver], rtol=tolerance) or row[solver] < row[other_solver]
                for other_solver in solvers
            ), axis=1)

    normalized_relative_scores = relative_scores.apply(normalize_row, axis=1)
    return normalized_relative_scores

def compare_solvers(solvers: List, problems: List[GraphV2Problem], loaded_datasets: dict):
    problem_types = set([problem.problem_type for problem in problems])
    mock_synapses = [GraphV2Synapse(problem=problem) for problem in problems]
    # results = {solver.__class__.__name__: [] for solver in solvers}
    run_times_dict = {solver.__class__.__name__: [] for solver in solvers}
    scores_dict = {solver.__class__.__name__: [] for solver in solvers}
    for i, solver in enumerate(solvers):
        run_times = []
        scores = []
        print(f"Running Solver {i+1} - {solver.__class__.__name__}")
        for mock_synapse in tqdm.tqdm(mock_synapses, desc=f"{solver.__class__.__name__} solving {problem_types}"):
            # generate the edges adhoc
            MetricTSPV2Generator.recreate_edges(problem = mock_synapse.problem, loaded_datasets=loaded_datasets)
            start_time = time.perf_counter()
            mock_synapse.solution = asyncio.run(solver.solve_problem(mock_synapse.problem))

            # remove edges and nodes to reduce memory consumption
            run_time = time.perf_counter() - start_time
            run_times.append(run_time)
            scores.append(get_tour_distance(mock_synapse))
            mock_synapse.problem.edges = None
            mock_synapse.problem.nodes = None
        run_times_dict[solver.__class__.__name__] = run_times
        scores_dict[solver.__class__.__name__] = scores
    return run_times_dict, scores_dict

def main():
    """Main evaluation function"""
    if not os.path.exists(os.path.join(ROOT_DIR, SAVE_DIR)):
        os.makedirs(os.path.join(ROOT_DIR, SAVE_DIR))

    # Create mock object for datasets
    class Mock:
        pass
    mock = Mock()
    load_default_dataset(mock)

    # Generate problems from existing results
    metric_problems, metric_sizes, distances, medians, worst = (
        MetricTSPV2Generator.generate_problems_from_existing_results(
            TOP_N, EXISTING_RESULTS_PATH, N_PROBLEMS
        )
    )
    distances = np.array(distances).transpose()

    # Define solvers to test
    test_solvers = [
        NearestNeighbourSolver(),
        OrtoolSolver(),
    ]

    # Compare solvers
    run_times_dict, scores_dict = compare_solvers(
        test_solvers, metric_problems, mock.loaded_datasets
    )

    # Create results DataFrame
    scores_df = pd.DataFrame(scores_dict)

    # Add top N distances
    for i in range(TOP_N):
        scores_df[f'top{i+1}'] = distances[i]

    scores_df['median'] = medians
    scores_df['worst'] = worst
    scores_df['problem_size'] = metric_sizes
    scores_df.index.name = 'problem_index'

    # Compute relative scores
    relative_scores_df = compute_relative_scores(scores_df)

    # Save results
    scores_df.to_csv(os.path.join(ROOT_DIR, SAVE_DIR, "solver_scores.csv"))
    relative_scores_df.to_csv(os.path.join(ROOT_DIR, SAVE_DIR, "solver_relative_scores.csv"))

    # Plot results
    average_relative_scores = relative_scores_df.groupby(scores_df['problem_size']).mean()
    average_relative_scores.plot(kind='bar', figsize=(12, 6))
    plt.title('Average Relative Score of Each Solver by Problem Size')
    plt.xlabel('Problem Size')
    plt.ylabel('Average Relative Score (Normalized)')
    plt.legend(title='Solver', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(os.path.join(ROOT_DIR, SAVE_DIR, "relative_score.png"))
    if can_show_plot():
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    main()
