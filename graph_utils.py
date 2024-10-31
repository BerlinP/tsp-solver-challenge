import numpy as np
from typing import List
from protocol import (
    GraphV2Problem,
)

REF_EARTH_RADIUS = 6378.388

def geom_edges(lat_lon_array: np.ndarray) -> np.ndarray:
    """Vectorized geometric distance calculation"""
    lat_lon_array = np.deg2rad(lat_lon_array)
    lat = lat_lon_array[:, 0]
    lon = lat_lon_array[:, 1]
    
    lat1, lat2 = np.meshgrid(lat, lat)
    lon1, lon2 = np.meshgrid(lon, lon)
    
    q1 = np.cos(lon1 - lon2)
    q2 = np.cos(lat1 - lat2)
    q3 = np.cos(lat1 + lat2)
    
    distances = REF_EARTH_RADIUS * np.arccos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0
    np.fill_diagonal(distances, 0)
    return distances

def man_2d_edges(lat_lon_array: np.ndarray) -> np.ndarray:
    """Vectorized Manhattan distance calculation"""
    lat = lat_lon_array[:, 0]
    lon = lat_lon_array[:, 1]
    
    lat1, lat2 = np.meshgrid(lat, lat)
    lon1, lon2 = np.meshgrid(lon, lon)
    
    distances = np.abs(lat1 - lat2) + np.abs(lon1 - lon2)
    np.fill_diagonal(distances, 0)
    return distances

def euc_2d_edges(lat_lon_array: np.ndarray) -> np.ndarray:
    """Vectorized Euclidean distance calculation"""
    lat = lat_lon_array[:, 0]
    lon = lat_lon_array[:, 1]
    
    lat1, lat2 = np.meshgrid(lat, lat)
    lon1, lon2 = np.meshgrid(lon, lon)
    
    distances = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
    np.fill_diagonal(distances, 0)
    return distances

def is_valid_path(path:List[int])->bool:
    # a valid path should have at least 3 return values and return to the source
    return (len(path)>=3) and (path[0]==path[-1])

def valid_problem(problem:GraphV2Problem)->bool:
    if problem.problem_type == 'Metric TSP':
        if (problem.directed==False) and (problem.visit_all==True) and (problem.to_origin==True) and (problem.objective_function=='min'):
            return True
        else:
            print(f"Received an invalid Metric TSP problem")
            print(problem.get_info(verbosity=2))
            return False
        
    elif problem.problem_type == 'General TSP':
        if (problem.directed==True) and (problem.visit_all==True) and (problem.to_origin==True) and (problem.objective_function=='min'):
            return True
        else:
            print(f"Received an invalid General TSP problem")
            print(problem.get_info(verbosity=2))
            return False
    
    elif problem.problem_type == 'Metric mTSP':
        if (problem.directed==False) \
            and (problem.visit_all==True) \
            and (problem.to_origin==True) \
            and (problem.objective_function=='min') \
            and (problem.n_salesmen > 1) \
            and (len(problem.depots)==problem.n_salesmen):
            if problem.single_depot == True:
                # assert that all the depots be at source city #0
                return True if all([depot==0 for depot in problem.depots]) else False
            else:
                # assert that all depots are different
                return True if len(set(problem.depots)) == len(problem.depots) else False
        else:
            print(f"Received an invalid Metric mTSP problem")
            print(problem.get_info(verbosity=2))
            return False
    elif problem.problem_type == 'General mTSP':
        if (problem.directed==True) \
            and (problem.visit_all==True) \
            and (problem.to_origin==True) \
            and (problem.objective_function=='min') \
            and (problem.n_salesmen > 1) \
            and (len(problem.depots)==problem.n_salesmen):
            if problem.single_depot == True:
                # assert that all the depots be at source city #0
                return True if all([depot==0 for depot in problem.depots]) else False
            else:
                # assert that all depots are different
                return True if len(set(problem.depots)) == len(problem.depots) else False
        else:
            print(f"Received an invalid General mTSP problem")
            print(problem.get_info(verbosity=2))
            return False