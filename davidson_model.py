import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class DavidsonBT:
    """
    Davidson extension of Bradley-Terry model to handle ties.
    """

    def __init__(self):
        self.strengths_ = None
        self.tie_param_ = None
        self.covariance_ = None

    @staticmethod
    def build_matrices(comparisons, n_items=None):
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
            jac=False,
            hess="2-point",
        )
        if not result.success:
            raise RuntimeError("Optimization failed: " + result.message)

        opt = result.x
        hess_inv = (
            result.hess_inv.todense()
            if hasattr(result.hess_inv, "todense")
            else np.linalg.inv(result.hess)
        )
        self.covariance_ = hess_inv

        pi = np.exp(opt[:n])
        pi /= pi.sum()
        nu = np.exp(opt[n])

        self.strengths_ = pi
        self.tie_param_ = nu
        self._theta_ = opt[:n]  # for CI and p-value use
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

    def get_confidence_intervals(self, conf_level=0.5):
        z = norm.ppf(1 - (1 - conf_level) / 2)
        std_errors = np.sqrt(np.diag(self.covariance_)[: len(self.strengths_)])
        intervals = []
        for theta, se in zip(self._theta_, std_errors):
            low = np.exp(theta - z * se)
            high = np.exp(theta + z * se)
            intervals.append((low, np.exp(theta), high))

        return intervals

    def pairwise_p_values(self):
        n = len(self._theta_)
        cov = self.covariance_[:n, :n]
        p_values = {}
        for i in range(n):
            for j in range(i + 1, n):
                diff = self._theta_[i] - self._theta_[j]
                se = np.sqrt(cov[i, i] + cov[j, j] - 2 * cov[i, j])
                z = diff / se
                p = 2 * (1 - norm.cdf(abs(z)))
                p_values[(i, j)] = p
        return p_values

    def pairwise_p_values_full(self):
        """
        Returns { (i, j): p } for all i != j.
        """
        basic = self.pairwise_p_values()  # only i<j
        full = {}
        n = len(self._theta_)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if i < j:
                    full[(i, j)] = basic[(i, j)]
                else:
                    # symmetry: p(i,j) == p(j,i)
                    full[(i, j)] = basic[(j, i)]
        return full

    def rank_pairwise_by_entropy(self):
        n = len(self.strengths_)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2, ptie = self.predict_proba(i, j)
                ent = -(
                    p1 * np.log(p1 + 1e-12)
                    + p2 * np.log(p2 + 1e-12)
                    + ptie * np.log(ptie + 1e-12)
                )
                pairs.append((i, j, ent))
        return sorted(pairs, key=lambda x: -x[2])


if __name__ == "__main__":

    def approx_equal(a, b, tol=1e-6):
        return abs(a - b) < tol

    print("Running DavidsonBT unit tests...")

    # Test build_matrices
    print("Test build_matrices...")
    comps = [(0, 1, "win1"), (1, 2, "tie"), (2, 0, "win2")]
    win, tie = DavidsonBT.build_matrices(comps, n_items=3)
    assert win.shape == (3, 3)
    assert tie.shape == (3, 3)
    assert win[0, 1] == 1
    assert win[0, 2] == 1  # from win2 on (2,0)
    assert tie[1, 2] == 1
    print(" build_matrices PASS")

    # Test from_comparisons and predict_proba sum-to-one
    print("Test from_comparisons and predict_proba...")
    # no comparisons => uniform strengths, nu=1
    empty_model = DavidsonBT.from_comparisons([], n_items=3)
    p01 = empty_model.predict_proba(0, 1)
    total = sum(p01)
    assert approx_equal(total, 1.0)
    # symmetry
    p10 = empty_model.predict_proba(1, 0)
    assert approx_equal(p01[0], p10[1])
    print(" from_comparisons and predict_proba PASS")

    # Test pairwise_p_values_full symmetry and keys
    print("Test pairwise_p_values_full...")
    # create trivial model where theta equal => p-values = 1
    trivial = empty_model
    pfull = trivial.pairwise_p_values_full()
    for (i, j), p in pfull.items():
        assert 0 <= p <= 1
        assert approx_equal(pfull[(j, i)], p)
    expected_pairs = [(i, j) for i in range(3) for j in range(3) if i != j]
    assert set(pfull.keys()) == set(expected_pairs)
    print(" pairwise_p_values_full PASS")

    # Test rank_pairwise_by_entropy ordering and count
    print("Test rank_pairwise_by_entropy...")
    ent_pairs = empty_model.rank_pairwise_by_entropy()
    # for 3 items, expect 3 pairs
    assert len(ent_pairs) == 3
    # entropies should be equal for uniform model
    ents = [e for (_, _, e) in ent_pairs]
    assert all(approx_equal(ents[0], e) for e in ents)
    # sorted descending
    assert ents == sorted(ents, reverse=True)
    print(" rank_pairwise_by_entropy PASS")

    print("All tests passed!")
