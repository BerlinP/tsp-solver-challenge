# TSP Solver Challenge

This repository contains a framework for implementing and evaluating Traveling Salesman Problem (TSP) solvers. The goal is to find the shortest possible path that visits each city exactly once and returns to the starting city.

## Getting Started

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Repository Structure

```
solver_evaluation/
├── solvers.py             # Solver implementations
├── evaluation_utils.py     # Utility functions and classes
├── evaluate_existing_results.py  # Main evaluation script
└── requirements.txt       # Project dependencies
```

## Implementing Your Solver

1. Create your solver by subclassing `BaseSolver` in `solvers.py`. Use `NearestNeighborSolver` as a reference:

```python
from typing import List
from .evaluation_utils import GraphV2Problem, Solution
from .solvers import BaseSolver

class YourSolver(BaseSolver):
    def problem_transformations(self, problem: GraphV2Problem):
        """Transform problem into format needed by your solver"""
        return problem.edges  # Default transformation returns edge matrix

    async def solve(self, formatted_problem, future_id: int) -> List[int]:
        """Implement your solving logic here"""
        # formatted_problem contains the edge matrix
        # Return a list of indices representing the tour
        # Example: [0, 2, 1, 3, 0] for a 4-city problem
        pass
```

### Key Components

- `problem_transformations`: Transforms the problem into a format your solver can use
  - Input: GraphV2Problem with nodes and edge distances
  - Output: Any format your solver needs

- `solve`: Core solving logic
  - Input: Transformed problem and future_id (for timeout tracking)
  - Output: List of indices representing the tour

### Example: NearestNeighborSolver

```python
class NearestNeighborSolver(BaseSolver):
    def problem_transformations(self, problem: GraphV2Problem):
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

        # Visit nearest unvisited node
        for _ in range(n - 1):
            if self.future_tracker.get(future_id):
                return None
                
            nearest_distance = np.inf
            nearest_node = None
            
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
```

## Evaluating Your Solver

1. Add your solver to `evaluate_existing_results.py`:

```python
from solvers import YourSolver

# Add your solver to test_solvers
test_solvers = [
    NearestNeighbourSolver(),
    YourSolver(),
    # Add more solvers...
]
```

2. Run the evaluation:
```bash
python evaluate_existing_results.py
```

3. Check results in `evaluation_results/`:
- `solver_scores.csv`: Raw performance scores
- `solver_relative_scores.csv`: Normalized relative performance
- `relative_score.png`: Performance visualization

## Evaluation Metrics

The framework evaluates solvers on:
1. Solution Quality: Total path distance (lower is better)
2. Runtime Performance: Time taken to find solution
3. Relative Performance: How well your solver performs compared to others

## Tips for Improvement

1. Study the problem structure in `GraphV2Problem`
2. Analyze how NearestNeighborSolver works
3. Consider these improvement areas:
   - Better initial city selection
   - Look-ahead strategies
   - Local optimization
   - Clustering for large problems
   - Meta-heuristics (e.g., 2-opt, 3-opt)

## Contributing

1. Fork the repository
2. Create your solver implementation
3. Test thoroughly
4. Submit a pull request with your results
