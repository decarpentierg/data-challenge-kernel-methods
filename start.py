import os

import numpy as np
import pickle
import pandas as pd

from molecules import Molecule
from kernel_methods import SVM
from auc import auc_score

from compute_kernel_matrices import compute_kernel_matrices
from download_kernel_matrices import download_wlsk, download_wlsk_test

# ----------------------------------------
# ENSURE THAT KERNEL MATRICES ARE COMPUTED
# ----------------------------------------

# Turn to False to recompute kernel matrices
# /!\ The computation is very long /!\
DOWNLOAD_KERNEL_MATRICES = True

# If kernel matrix is not already computed/downloaded, compute or download it
if not os.path.isfile("kernels/wlsk.npy"):
    if DOWNLOAD_KERNEL_MATRICES:
        download_wlsk()
    else:
        compute_kernel_matrices("train", 0, 6000, "train", 0, 6000, "wlsk.npy")

if not os.path.isfile("kernels/wlsk_test.npy"):
    if DOWNLOAD_KERNEL_MATRICES:
        download_wlsk_test()
    else:
        compute_kernel_matrices("train", 0, 6000, "test", 0, 2000, "wlsk_test.npy")

# ---------------------------
# LOAD TRAINING AND TEST DATA
# ---------------------------

with open("data/training_data.pkl", "rb") as file:
    training_data = pickle.load(file)

with open("data/training_labels.pkl", "rb") as file:
    training_labels = 2 * pickle.load(file) - 1  # labels in {-1, 1}

with open("data/test_data.pkl", "rb") as file:
    test_data = pickle.load(file)

training_data = [Molecule(graph) for graph in training_data]
test_data = [Molecule(graph) for graph in test_data]

K = np.load("kernels/wlsk.npy")
K_test = np.load("kernels/wlsk_test.npy")

# -------
# FIT SVC
# -------

np.random.seed(44)

svc = SVM(C=1.0)
print("Fitting SVC...")
svc.fit(K, training_labels)

# print score on training data
scores_train = svc.decision_function(K)
print(
    "AUC score on training data:", auc_score(training_labels, svc.decision_function(K))
)

# -----------------------------
# Make predictions on test data
# -----------------------------

# predict labels on test data
pred = svc.decision_function(K_test.T)

Yte = {"Predicted": pred}
dataframe = pd.DataFrame(Yte)
dataframe.index += 1
dataframe.to_csv("test_pred_3.csv", index_label="Id")
