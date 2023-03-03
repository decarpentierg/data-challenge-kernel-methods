import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import pickle
import argparse
import itertools

from molecules import Molecule
from kernels import wlsk

# Parse arguments
parser = argparse.ArgumentParser(description="Compute kernel")
parser.add_argument(
    "indices", 
    type=str, 
    help="Compute kernel between entries %2 to %3 of %1-set and entries %5 to %6 of %4-set where argument=%1,%2,%3,%4,%5,%6"
)
args = parser.parse_args()
set1, start1, stop1, set2, start2, stop2 = args.indices.split(",")
start1, stop1, start2, stop2 = [int(x) for x in [start1, stop1, start2, stop2]]

# Load data
with open("data/training_data.pkl", "rb") as file:
    training_data = pickle.load(file)
training_data = [Molecule(graph) for graph in training_data]

with open("data/test_data.pkl", "rb") as file:
    test_data = pickle.load(file)
test_data = [Molecule(graph) for graph in test_data]

def dataset(name):
    if name == 'train':
        return training_data
    else:
        return test_data

data1, data2 = dataset(set1), dataset(set2)

# Define process map

def process(indices_pair):
    idx1, idx2 = indices_pair
    return wlsk(data1[idx1], data2[idx2])

# Compute kernels

indices = list(itertools.product(range(start1, stop1), range(start2, stop2)))
kernel_values = process_map(process, indices, chunksize=100)

np.savez_compressed(f'kernels/{args.indices}', indices=indices, kernel_values=kernel_values)
