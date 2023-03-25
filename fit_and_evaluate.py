import numpy as np
import pickle

from kernel_methods import SVM

with open("data/training_labels.pkl", "rb") as file:
    training_labels = 2 * pickle.load(file) - 1  # labels in {-1, 1}

K = np.load('kernels/wlsk.npy')
K_test = np.load('kernels/wlsk_test.npy')

np.random.seed(42)
idx = np.random.permutation(6000)

K_train = K[idx[:5000],:][:,idx[:5000]]
K_eval = K[idx[5000:],:][:,idx[:5000]]

labels_train = training_labels[idx[:5000]]
labels_eval = training_labels[idx[5000:]]

svc = SVM(C=1)  # change index here to change type of SVC
svc.fit(K_train, labels_train)
