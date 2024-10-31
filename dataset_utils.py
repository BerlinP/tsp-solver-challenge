import torch
import tempfile
import requests
from requests import HTTPError
import os
from pympler import asizeof
import io
import gzip
import osmium
from pathlib import Path
import numpy as np
import hashlib
import time
from huggingface_hub import hf_hub_download

ASIA_MSB_DETAILS = {"endpoint":"https://download.geofabrik.de/asia/malaysia-singapore-brunei-240905.osm.pbf", 
                    "ref_id":"Asia_MSB"}
WORLD_TSP_DETAILS = {"endpoint":"https://www.math.uwaterloo.ca/tsp/world/world.tsp.gz",
                     "ref_id":"World_TSP"}

DATASET_DIR = Path(__file__).resolve().parent.joinpath("dataset")

def create_directory_if_not_exists(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

def get_file_path(ref_id:str)->Path:
    '''
    Inputs: ref_id unique identifier of supported dataset

    returns the file paths of the zipped coordinates
    '''
    file_with_extension = ref_id + ".npz"
    return DATASET_DIR / file_with_extension

def check_and_get_msb():
    fp = get_file_path(ASIA_MSB_DETAILS['ref_id'])
    if fp.exists():
        # we have already downloaded and processed the data
        print(f"{ASIA_MSB_DETAILS['ref_id']} already downloaded")
        return
    else:
        try:
            create_directory_if_not_exists(DATASET_DIR)
            print(f"Downloading {ASIA_MSB_DETAILS['ref_id']} data from huggingface")
            hf_hub_download(repo_id="Graphite-AI/coordinate_data", filename="Asia_MSB.npz", repo_type="dataset", local_dir=DATASET_DIR)
        except:
            # Obtain byte content through get request to endpoint
            print(f"Downloading {ASIA_MSB_DETAILS['endpoint']} data from source")
            start_time = time.time()
            response = requests.get(ASIA_MSB_DETAILS['endpoint'])
            try:
                # check for response status
                response.raise_for_status()

                # create a tempfile to write the content to then read and process
                with tempfile.NamedTemporaryFile(delete=False, suffix='.osm.pbf') as temp_file:
                    temp_file_name = temp_file.name
                    
                    # Write byte content to the temporary file
                    temp_file.write(response.content)

                download_time = time.time() - start_time
                print(f"\nDownload completed in {download_time:.2f} seconds. Constructing coordinates from pbf file. This will take a few minutes...")

                # Define and Instantiate the osmium handler for extracting coordinate information
                class WayNodeHandler(osmium.SimpleHandler):
                    def __init__(self):
                        super(WayNodeHandler, self).__init__()
                        self.nodes = np.empty((28000000,3), dtype=np.float32) # we know the exact number of expected nodes
                        self.index = 0
                    
                    def node(self, n):
                        # Create a new record with the node's data
                        self.nodes[self.index,0] = n.id
                        self.nodes[self.index,1] = n.location.lat
                        self.nodes[self.index,2] = n.location.lon
                        self.index += 1

                handler = WayNodeHandler()
                handler.apply_file(temp_file_name)
                print(handler.index)

                print(f"{asizeof.asizeof(handler)} bytes")
                print(f"{asizeof.asizeof(handler.nodes[0])} bytes for a single node")

                array_bytes = np.array(handler.nodes).tobytes()
                hash_algo='md5'
                hash_func=getattr(hashlib, hash_algo)()
                hash_func.update(array_bytes)
                print(f"Coordinates extracted with corresponding MD5 hash: {hash_func.hexdigest()} with {handler.index} nodes")

                create_directory_if_not_exists(DATASET_DIR)
                np.savez_compressed( DATASET_DIR / (ASIA_MSB_DETAILS['ref_id']+'.npz'), data=np.array(handler.nodes[:handler.index]))
                print(f"{DATASET_DIR / ASIA_MSB_DETAILS['ref_id']} coordinates saved")

                # Clean up: Remove the temporary file
                if os.path.exists(temp_file_name):
                    os.remove(temp_file_name)

            except HTTPError as e:
                print(f"Error fetching data from endpoint: {e}")

def check_and_get_wtsp():
    fp = get_file_path(WORLD_TSP_DETAILS['ref_id'])
    if fp.exists():
        # we have already downloaded and processed the data
        print(f"{WORLD_TSP_DETAILS['ref_id']} already downloaded")
        return
    else:
        try:
            create_directory_if_not_exists(DATASET_DIR)
            print(f"Downloading {WORLD_TSP_DETAILS['ref_id']} data from huggingface")
            hf_hub_download(repo_id="Graphite-AI/coordinate_data", filename="World_TSP.npz", repo_type="dataset", local_dir=DATASET_DIR)
        except:
            print(f"Downloading {WORLD_TSP_DETAILS['ref_id']} data from source")
            # Obtain byte content through get request to endpoint
            start_time = time.time()
            response = requests.get(WORLD_TSP_DETAILS['endpoint'])
            try:
                # check for response status
                response.raise_for_status()

                download_time = time.time() - start_time
                print(f"\nDownload completed in {download_time:.2f} seconds. Constructing coordinates from tsp file.")

                # Decompress the response content if it is gzipped
                with io.BytesIO(response.content) as compressed_file:
                    with gzip.GzipFile(fileobj=compressed_file) as decompressed_file:
                        # Read the decompressed lines and decode each from bytes to string
                        lines = [line.decode('utf-8').replace("\n", "") for line in decompressed_file.readlines()]
                        print(f"{asizeof.asizeof(compressed_file)} bytes - Compressed")
                        print(f"{asizeof.asizeof(decompressed_file)} bytes - Decompressed")
                coordinates = []
                line_iter = iter(lines)
                item = next(line_iter)
                assert item == "NAME : world"
                is_coordinate = False
                try:
                    while item != "EOF":
                        if item == "NODE_COORD_SECTION":
                            is_coordinate = True
                            item = next(line_iter)
                            continue
                        if is_coordinate:
                            try:
                                idx, lat, lon = item.split(" ")
                                coordinates.append([int(idx), float(lat), float(lon)])
                            except ValueError:
                                print(f"Error unpacking tsp file, trying to map (index, lat, lon) from {item}")
                        item = next(line_iter)
                except StopIteration:
                    print(f"No EOF signal found while unpacking {WORLD_TSP_DETAILS['ref_id']} tsp file")

                array_bytes = np.array(coordinates).tobytes()
                hash_algo='md5'
                hash_func=getattr(hashlib, hash_algo)()
                hash_func.update(array_bytes)
                print(f"Coordinates extracted with corresponding MD5 hash: {hash_func.hexdigest()}")

                create_directory_if_not_exists(DATASET_DIR)
                np.savez_compressed(DATASET_DIR / (WORLD_TSP_DETAILS['ref_id']+'.npz'), data=np.array(coordinates))
                print(f"{DATASET_DIR / WORLD_TSP_DETAILS['ref_id']} coordinates saved")

            except HTTPError as e:
                print(f"Error fetching data from endpoint: {e}")

def download_default_datasets():
    check_and_get_msb()
    check_and_get_wtsp()

def load_default_dataset(neuron):
    '''
    Loads the default dataset into neuron as a dict of {"dataset_name":{"coordinates":np.array, "checksum":str}}
    '''
    create_directory_if_not_exists(DATASET_DIR)
    # check and process default datasets
    download_default_datasets()

    # set the neuron dataset attribute
    neuron.loaded_datasets = {
        ASIA_MSB_DETAILS['ref_id']: load_dataset(ASIA_MSB_DETAILS['ref_id']),
        WORLD_TSP_DETAILS['ref_id']: load_dataset(WORLD_TSP_DETAILS['ref_id'])
    }

def load_dataset(ref_id:str)->dict:
    '''
    loads in coordinate information from the referenced .npz file and returns the coordinates

    Inputs: ref_id name of dataset
    '''
    filepath = get_file_path(ref_id)
    try:
        array_map = np.load(filepath)
        return {"data":array_map['data'],"checksum": get_checksum(array_map['data'])}
    except OSError as e:
        print(f"Error while loading file: {filepath}. Check if file exists.")
        print(f"Full Error Message: {e}")

def get_checksum(coordinates:np.array)->str:
    '''
    Function getting 128 byte checksum for aligning datasets between validator and miner.

    Inputs: coordinates numpy array representing node coordinates of the dataset
    Output: md5 hash
    '''
    hash_algo='md5'
    hash_func=getattr(hashlib, hash_algo)()
    hash_func.update(coordinates.tobytes())
    return hash_func.hexdigest()