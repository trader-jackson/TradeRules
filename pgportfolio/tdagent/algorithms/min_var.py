from ..tdagent import TDAgent
import numpy as np
from scipy.optimize import minimize

class MinVar(TDAgent):
    """ Minimum Variance Portfolio strategy.
    This strategy aims to create a portfolio with the lowest possible risk (variance)
    regardless of expected returns.

    Reference:
        H. Markowitz. Portfolio Selection, 1952.
        The Journal of Finance, Vol. 7, No. 1, pp. 77-91
    """
    def __init__(self, window=30, b=None):
        """
        :param window: Window size for calculating covariance matrix
        :param b: Initial portfolio weights. Default is uniform.
        """
        super(MinVar, self).__init__()
        self.window = window
        self.b = b
        self.history_matrix = None

    def decide_by_history(self, x, last_b):
        """ Calculate new portfolio weights based on minimum variance optimization.
        :param x: input matrix with shape (1, window_size, coin_number+1)
        :param last_b: last portfolio weight vector
        """
        # Get the last relative price vector
        x = self.get_last_rpv(x)
        
        # Initialize b to uniform if not provided
        if self.b is None:
            self.b = np.ones(len(x)) / len(x)
        
        # Record history for covariance calculation
        if self.history_matrix is None:
            self.history_matrix = np.array([x])
        else:
            self.history_matrix = np.vstack((self.history_matrix, x))
            # Keep only the most recent window periods
            if len(self.history_matrix) > self.window:
                self.history_matrix = self.history_matrix[-self.window:]
        
        # Only optimize if we have enough history
        if len(self.history_matrix) >= 2:  # Need at least 2 periods to calculate covariance
            try:
                # Calculate optimal weights using minimum variance optimization
                self.b = self.optimize_min_variance(self.history_matrix)
            except Exception as e:
                # If optimization fails, keep the previous weights
                print(f"Optimization failed: {e}")
                pass
        
        return self.b
    
    def optimize_min_variance(self, returns):
        """ Find weights that minimize portfolio variance.
        :param returns: Matrix of historical returns
        :returns: Optimal portfolio weights
        """
        n_assets = returns.shape[1]
        
        # Calculate the covariance matrix
        cov_matrix = np.cov(returns.T)
        
        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets
        
        # Objective function: minimize portfolio variance
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: no short selling (weights between 0 and 1)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Minimize the objective function
        result = minimize(portfolio_variance, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        # Return the optimal weights
        return result.x