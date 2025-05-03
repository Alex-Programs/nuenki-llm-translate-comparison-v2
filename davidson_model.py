import numpy as np
from scipy.optimize import minimize


class DavidsonBT:
    """
    Davidson extension of Bradley-Terry model to handle ties.
    """

    def __init__(self):
        self.strengths_ = None
        self.tie_param_ = None

    @staticmethod
    def build_matrices(comparisons, n_items=None):
        """
        Build win and tie count matrices from raw comparisons.
        comparisons: iterable of (i, j, result) where result is "win1", "win2", or "tie"
        n_items: total number of items (optional)
        """
        # determine number of items
        if n_items is None:
            max_id = max(max(i, j) for i, j, _ in comparisons)
            n_items = max_id + 1
        win = np.zeros((n_items, n_items), int)
        tie = np.zeros((n_items, n_items), int)
        for i, j, result in comparisons:
            if result == "win1":
                win[i, j] += 1
            elif result == "win2":
                win[j, i] += 1
            elif result == "tie":
                a, b = (i, j) if i < j else (j, i)
                tie[a, b] += 1
            else:
                raise ValueError("result must be 'win1', 'win2', or 'tie'")
        return win, tie

    @classmethod
    def from_comparisons(
        cls, comparisons, n_items=None, init_strengths=None, init_tie=1.0
    ):
        win, tie = cls.build_matrices(comparisons, n_items)
        return cls().fit(win, tie, init_strengths, init_tie)

    def _neg_log_likelihood(self, params, win_matrix, tie_matrix):
        n = win_matrix.shape[0]
        theta = params[:n]
        tau = params[n]
        pi = np.exp(theta)
        nu = np.exp(tau)
        ll = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                denom = pi[i] + pi[j] + nu * np.sqrt(pi[i] * pi[j])
                if win_matrix[i, j] > 0:
                    ll += win_matrix[i, j] * np.log(pi[i] / denom)
                if win_matrix[j, i] > 0:
                    ll += win_matrix[j, i] * np.log(pi[j] / denom)
                if tie_matrix[i, j] > 0:
                    ll += tie_matrix[i, j] * np.log(
                        (nu * np.sqrt(pi[i] * pi[j])) / denom
                    )
        return -ll

    def fit(self, win_matrix, tie_matrix, init_strengths=None, init_tie=1.0):
        n = win_matrix.shape[0]
        theta0 = np.log(init_strengths) if init_strengths is not None else np.zeros(n)
        tau0 = np.log(init_tie)
        x0 = np.concatenate([theta0, [tau0]])
        result = minimize(
            fun=self._neg_log_likelihood,
            x0=x0,
            args=(win_matrix, tie_matrix),
            method="L-BFGS-B",
        )
        opt = result.x
        pi = np.exp(opt[:n])
        pi /= pi.sum()
        nu = np.exp(opt[n])
        self.strengths_ = pi
        self.tie_param_ = nu
        return self

    def predict_proba(self, i, j):
        pi_i = self.strengths_[i]
        pi_j = self.strengths_[j]
        nu = self.tie_param_
        denom = pi_i + pi_j + nu * np.sqrt(pi_i * pi_j)
        return pi_i / denom, pi_j / denom, (nu * np.sqrt(pi_i * pi_j)) / denom

    def get_strengths(self):
        return self.strengths_

    def get_tie_param(self):
        return self.tie_param_


# example of helper functions
# comparisons = [
#     (0, 1, "win1"),
#     (1, 2, "tie"),
#     (2, 0, "win2"),
# ]
# model = DavidsonBT.from_comparisons(comparisons)
# print(model.get_strengths(), model.get_tie_param())
