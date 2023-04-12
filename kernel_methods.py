import numpy as np


class SVM():
    """Kernel SVC fitted with SMO.
    Class for Support Vector Classifiers (SVC) fitted with Sequential Minimal Optimization (SMO).
    The classifying function is defined as
        x -> sign(f(x) + b)
    where f(x) = sum_i alpha_i * y_i * kernel(x_i, x)
    The parameters to fit on the data are the $alpha_i$ and $b$.
    
    Attributes
    ----------    
    C: float
        margin penalization parameter (inverse of a regularization parameter)
    
    alpha: ndarray of shape (n_samples,)
        coefficients alpha_i for the support vectors x_i
    
    b: float
        offset of the classifier
    
    y: ndarray of shape (n_samples,)
        labels of the training data
    
    epsilon: float
        tolerance for the comparison of floats
    
    tol: float
        tolerance for the comparison of floats
    
    max_iter: int
        maximum number of iterations in the SMO optimization
    """
    def __init__(self, C=1.0, epsilon=1e-5, tol=1e-3, max_iter=1000):
        """Class constructor
        
        Arguments
        ---------
        See class attributes.
        """
        self.C = C
        self.epsilon = epsilon
        self.tol = tol
        self.max_iter = max_iter
        
    def fit(self, K, y):
        """Fit the classifier.

        Arguments
        ---------
        K: ndarray of shape (n_samples, n_samples)
            kernel matrix of training data
        
        y: ndarray of shape (n_samples)
            training labels
        """
        n_samples = K.shape[0]
        alpha = np.zeros(n_samples)
        b = 0
        eta = 0
        L = 0
        H = 0
        E = np.zeros(n_samples)
        for iteration in range(self.max_iter):
            num_changed_alphas = 0
            for i in range(n_samples):
                E[i] = b + np.sum(alpha * y * K[i]) - y[i]
                if ((y[i] * E[i] < -self.tol and alpha[i] < self.C) or
                    (y[i] * E[i] > self.tol and alpha[i] > 0)):
                    j = np.random.choice(np.delete(np.arange(n_samples), i))
                    E[j] = b + np.sum(alpha * y * K[j]) - y[j]
                    alpha_i_old, alpha_j_old = alpha[i], alpha[j]
                    if y[i] != y[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])
                    if L == H:
                        continue
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    alpha[j] = alpha[j] - y[j] * (E[i] - E[j]) / eta
                    alpha[j] = min(H, alpha[j])
                    alpha[j] = max(L, alpha[j])
                    if abs(alpha[j] - alpha_j_old) < self.epsilon:
                        continue
                    alpha[i] = alpha[i] + y[i] * y[j] * (alpha_j_old - alpha[j])
                    b1 = b - E[i] - y[i] * (alpha[i] - alpha_i_old) * K[i, i] - y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                    b2 = b - E[j] - y[i] * (alpha[i] - alpha_i_old) * K[i, j] - y[j] * (alpha[j] - alpha_j_old) * K[j, j]
                    if 0 < alpha[i] < self.C:
                        b = b1
                    elif 0 < alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
                    num_changed_alphas += 1
            if num_changed_alphas == 0:
                break
        self.alpha = alpha
        self.b = b
        self.y = y
    
    def decision_function(self, K_test):
        """Decision function: its sign is the predicted class.

        Arguments
        ---------
        K_test: ndarray of shape (n_test_samples, n_samples)
            kernel matrix of test vs training data
        
        Returns
        -------
        ndarray of shape (n_test_samples)
            array of decision logits for each test sample
        """
        return np.sum(self.alpha * self.y * K_test, axis=1) + self.b
        
    def predict(self, K_test):
        """Predict labels for test data
        
        Arguments
        ---------
        K_test: ndarray of shape (n_test_samples, n_samples)
            kernel matrix of test vs training data
        
        Returns
        -------
        ndarray of shape (n_test_samples)
            array of label predictions for each test sample
        """
        return np.sign(self.decision_function(K_test))
