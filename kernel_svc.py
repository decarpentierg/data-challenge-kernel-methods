import numpy as np
from scipy import optimize


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
    
    def fit(self, X, y):
        """Fit the model. 
        
        Parameters
        ----------
        X: ndarray of shape (N, d)
            data points
        y: ndarray of shape (N,)
            labels (-1 or 1)
        """
        # ---------------------------
        # Define optimization problem
        # ---------------------------

        N = len(y)
        K = self.kernel(X, X)
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
        self.support = X[alpha * y > self.epsilon]
        self.alpha = alpha[alpha * y > self.epsilon]
        
        # offset
        f = K @ alpha
        on_margin = np.logical_and(alpha * y > self.epsilon, alpha * y < self.C - self.epsilon)
        self.b = np.median(y[on_margin] - f[on_margin])
    
    def predict(self, X):
        """Predict label values in {-1, 1} for new data points (after model has been fitted).
        
        Parameters
        ----------
        X: ndarray of shape (N, d)
            data points
        
        Returns: ndarray of shape (N,)
            predicted labels
        """
        K = self.kernel(self.support, X)
        return 2 * (self.alpha @ K + self.b > 0) - 1
