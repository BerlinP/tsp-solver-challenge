from typing import List, Tuple, Union, Literal, Iterable
from dataclasses import dataclass
from pydantic import BaseModel, Field, model_validator, conint, confloat


class GraphV2Problem(BaseModel):
    """Problem representation for TSP"""
    problem_type: Literal['Metric TSP', 'General TSP'] = Field('Metric TSP', description="Problem Type")
    objective_function: str = Field('min', description="Objective Function")
    visit_all: bool = Field(True, description="Visit All Nodes")
    to_origin: bool = Field(True, description="Return to Origin")
    n_nodes: conint(ge=2000, le=5000) = Field(2000, description="Number of Nodes (must be between 2000 and 5000)")
    selected_ids: List[int] = Field(default_factory=list, description="List of selected node positional indexes")
    cost_function: Literal['Geom', 'Euclidean2D', 'Manhatten2D', 'Euclidean3D', 'Manhatten3D'] = Field('Geom', description="Cost function")
    dataset_ref: Literal['Asia_MSB', 'World_TSP'] = Field('Asia_MSB', description="Dataset reference file")
    nodes: Union[List[List[Union[conint(ge=0), confloat(ge=0)]]], Iterable, None] = Field(default_factory=list, description="Node Coordinates")
    edges: Union[List[List[Union[conint(ge=0), confloat(ge=0)]]], Iterable, None] = Field(default_factory=list, description="Edge Weights")
    directed: bool = Field(False, description="Directed Graph")
    simple: bool = Field(True, description="Simple Graph")
    weighted: bool = Field(False, description="Weighted Graph")
    repeating: bool = Field(False, description="Allow Repeating Nodes")

    @model_validator(mode='after')
    def force_obj_function(self):
        if self.problem_type in ['Metric TSP', 'General TSP']:
            assert self.objective_function == 'min', ValueError('Currently only supports minimization TSP')
        return self

    def get_info(self, verbosity: int = 1) -> dict:
        """Get problem information at different verbosity levels"""
        info = {}
        if verbosity == 1:
            info["Problem Type"] = self.problem_type
        elif verbosity == 2:
            info["Problem Type"] = self.problem_type
            info["Objective Function"] = self.objective_function
            info["To Visit All Nodes"] = self.visit_all
            info["To Return to Origin"] = self.to_origin
            info["Number of Nodes"] = self.n_nodes
            info["Directed"] = self.directed
            info["Simple"] = self.simple
            info["Weighted"] = self.weighted
            info["Repeating"] = self.repeating
        elif verbosity == 3:
            for field in self.model_fields:
                description = self.model_fields[field].description
                value = getattr(self, field)
                info[description] = value
        return info

@dataclass
class Solution:
    """Solution representation"""
    tour: List[int]

class GraphV2Synapse:
    """Container for problem and solution"""
    def __init__(self, problem: GraphV2Problem):
        self.problem = problem
        self.solution = None