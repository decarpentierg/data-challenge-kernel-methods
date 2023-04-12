import numpy as np
import pickle
import argparse

from molecules import Molecule
from kernels import wlsk

def compute_kernel_matrices(set1, start1, stop1, set2, start2, stop2, filename):
    """Compute kernel between entries from start1 to stop1 of dataset set1 and entries from start2 to stop2 of dataset set2.
    Save the result in results/{filename}.npy.
    The function tqdm.contrib.concurrent.process_map is used for multiprocessing to accelerate the computations.

    /!\ Computing all 36M elements of the training kernel matrix take approximately 20h on a single CPU. /!\
    """

    # Load data
    with open("data/training_data.pkl", "rb") as file:
        training_data = pickle.load(file)
    training_data = [Molecule(graph) for graph in training_data]

    with open("data/test_data.pkl", "rb") as file:
        test_data = pickle.load(file)
    test_data = [Molecule(graph) for graph in test_data]

    def dataset(name):
        if name == "train":
            return training_data
        else:
            return test_data

    data1, data2 = dataset(set1), dataset(set2)

    kernel_values = [[wlsk(data1[idx1], data2[idx2]) for idx2 in range(start2, stop2)] for idx1 in range(start1, stop1)]
    kernel_values = np.array(kernel_values, dtype=np.int32)

    np.save(f"kernels/{filename}", kernel_values)

if __name__ == "__main__":
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

    compute_kernel_matrices(set1, start1, stop1, set2, start2, stop2, filename=args.indices)
