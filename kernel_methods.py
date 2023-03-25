import numpy as np
from scipy import optimize
from tqdm import tqdm
from numba import int32, float32
from numba.experimental import jitclass


class KernelSVC:
    """Class for Support Vector Classifiers.
    The classifying function is defined as
        x -> sign(f(x) + b)
    where f(x) = sum_i alpha_i * kernel(x_i, x)
    The parameters to fit on the data are the $alpha_i$ and $b$.
    
    Attributes
    ----------
    kernel: function, signature = ndarray, ndarray -> ndarray
        kernel to use
    
    C: float
        margin penalization parameter (inverse of a regularization parameter)

    support: ndarray of shape (n_support_vectors, d)
        support vectors
    
    alpha: ndarray of shape (n_support_vectors,)
        coefficients alpha_i for the support vectors x_i
    
    b: float
        offset of the classifier
    
    epsilon: float
        tolerance for the comparison of floats
    """
    
    def __init__(self, C, kernel, epsilon=1e-3):
        """Class constructor. See class docstring for parameter descriptions."""
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
    
    def fit(self, y):
        """Fit the model. 
        
        Parameters
        ----------
        y: ndarray of shape (N,)
            labels (-1 or 1)
        """
        # ---------------------------
        # Define optimization problem
        # ---------------------------

        N = len(y)
        K = self.kernel
        JAC_INEQ = np.ones((2 * N, N))  # Jacobian for the inequality constraints
        JAC_INEQ[:N] = np.diag(y)
        JAC_INEQ[N:] = -np.diag(y)

        # Lagrange dual problem loss
        def loss(alpha):
            return -np.sum(alpha * y) + 0.5 * alpha @ K @ alpha

        # Partial derivative of the dual loss wrt alpha
        def grad_loss(alpha):
            return -y + K @ alpha
            
        # Constraints on alpha are expressed as:
        # fun_eq(alpha)  = 0
        # fun_ineq(alpha) >= 0
        
        # Equality constraint
        fun_eq = lambda alpha: np.sum(alpha)
        jac_eq = lambda alpha: np.ones(N)
        # Inequality constraints
        fun_ineq = lambda alpha: np.concatenate((alpha * y, -alpha * y + self.C))
        jac_ineq = lambda alpha: JAC_INEQ
        
        constraints = ({'type': 'eq',   'fun': fun_eq,   'jac': jac_eq},
                       {'type': 'ineq', 'fun': fun_ineq, 'jac': jac_ineq})
        
        # --------
        # Optimize
        # --------

        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N),
                                   method='SLSQP',
                                   jac=lambda alpha: grad_loss(alpha),
                                   constraints=constraints)
        alpha = optRes.x

        # -----------------
        # Update attributes
        # -----------------

        # support vectors and coefficients
        self.support = alpha * y > self.epsilon
        self.alpha = alpha[alpha * y > self.epsilon]
        
        # offset
        f = K @ alpha
        on_margin = np.logical_and(alpha * y > self.epsilon, alpha * y < self.C - self.epsilon)
        self.b = np.median(y[on_margin] - f[on_margin])
    
    def predict(self, kernel):
        """Predict label values in {-1, 1} for new data points (after model has been fitted).
        
        Parameters
        ----------
        X: ndarray of shape (N, d)
            data points
        
        Returns: ndarray of shape (N,)
            predicted labels
        """
        K = kernel
        return 2 * (self.alpha @ K + self.b > 0) - 1


class KernelPCA:
    
    def __init__(self, kernel, r=2):                             
        self.kernel = kernel
        self.alpha = None  # Matrix of shape (N, r) representing the top r eingenvectors
        self.lmbda = None  # Vector of size r representing the top r eingenvalues
        self.r = r  # Number of principal components
        
    def compute_PCA(self):
        
        # Compute Gram matrix and center it
        N = self.kernel.shape[0]
        center = np.eye(N) - np.ones((N, N)) / N # centering matrix
        K = self.kernel  # shape (N, N), Gram matrix
        Kc = center @ K @ center  # center Gram matrix
        
        # Compute top r eigenvalues
        w, v = np.linalg.eig(Kc)  # shapes (N,), (N, N)
        w, v = w.real, v.real
        idx = np.argsort(w)[::-1][:self.r]  # indices of r largest eigenvalues
        
        # Assign the vectors
        self.lmbda = w[idx]  # shapes (r,)
        self.alpha = v[:, idx]  # shapes (N, r)
        
    def transform(self, kernel):
        # Input : matrix x of shape (M, d)
        # Output: matrix of shape(M, r)
        
        K1 = self.kernel  # shape (N, N)
        K2 = kernel # shape (M, N)
        
        K = K2 - np.mean(K2, axis=1, keepdims=True) - np.mean(K1, axis=0) + np.mean(K1) 
        
        return K @ (self.alpha / np.sqrt(self.lmbda))


spec = [
    ('C', float32),
    ('epsilon', float32),
    ('tol', float32),
    ('max_iter', int32),
    ('alpha', float32[:]),
    ('b', float32[:]),
    ('y', int32[:])
]

# @jitclass(spec)
class SVM():
    """SVC fitted with SMO."""
    def __init__(self, C=1.0, epsilon=1e-5, tol=1e-3, max_iter=1000):
        self.C = C
        self.epsilon = epsilon
        self.tol = tol
        self.max_iter = max_iter
        
    def fit(self, K, y):
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
        return np.sum(self.alpha * self.y * K_test, axis=1) + self.b
        
    def predict(self, K_test):
        return np.sign(self.decision_function(K_test))
