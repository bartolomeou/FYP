from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import gennorm

rng = np.random.default_rng()


class TargetDistribution(ABC):
    def __init__(self, n_var):
        self.n_var = n_var

    @abstractmethod
    def logpi(self, x):
        pass

    @abstractmethod
    def d1_logpi(self, x):
        pass

    @abstractmethod
    def d2_logpi(self, x):
        pass

    @abstractmethod
    def d3_logpi(self, x):
        pass

    @abstractmethod
    def direct_sample(self, size):
        pass

    def get_var_labels(self):
        labels = []

        for i in range(self.n_var):
            labels.append(f"X_{{i}}")

        return labels


class GeneralNormal(TargetDistribution):
    def __init__(self, n_var=1, mu=0, alpha=1, beta=2):
        super().__init__(n_var)
        self.mu = mu
        self.alpha = alpha
        self.beta = beta

    def logpi(self, x):
        return -((np.abs(x - self.mu) / self.alpha) ** self.beta)

    def d1_logpi(self, x):
        diff = x - self.mu
        return (
            -(self.beta / self.alpha**self.beta)
            * np.abs(diff) ** (self.beta - 1)
            * np.sign(diff)
        )

    def d2_logpi(self, x):
        diff = x - self.mu
        return -((self.beta * (self.beta - 1)) / self.alpha**self.beta) * np.abs(
            diff
        ) ** (self.beta - 2)

    def d3_logpi(self, x):
        diff = x - self.mu
        return (
            -((self.beta * (self.beta - 1) * (self.beta - 2)) / self.alpha**self.beta)
            * np.abs(diff) ** (self.beta - 3)
            * np.sign(diff)
        )

    def direct_sample(self, size):
        return gennorm.rvs(beta=self.beta, loc=self.mu, scale=self.alpha, size=size)


class SmoothGeneralNormal(TargetDistribution):
    def __init__(self, n_var=1, mu=0, alpha=1, beta=2, epsilon=1):
        super().__init__(n_var)
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def logpi(self, x):
        return -(
            (np.sqrt(self.epsilon * +((x - self.mu) ** 2)) / self.alpha) ** self.beta
        )

    def d1_logpi(self, x):
        diff = x - self.mu
        return (
            -(self.beta / self.alpha**self.beta)
            * diff
            * (self.epsilon + diff**2) ** ((self.beta - 2) / 2)
        )

    def d2_logpi(self, x):
        diff = x - self.mu
        return (
            -(self.beta / self.alpha**self.beta)
            * (self.epsilon + diff**2) ** ((self.beta - 4) / 2)
            * ((self.beta - 1) * diff**2 + self.epsilon)
        )

    def d3_logpi(self, x):
        diff = x - self.mu
        return (
            -((self.beta * (self.beta - 2)) / self.alpha**self.beta)
            * diff
            * (self.epsilon + diff**2) ** ((self.beta - 6) / 2)
            * ((self.beta - 1) * diff**2 + 3 * self.epsilon)
        )

    def direct_sample(self, size):
        return gennorm.rvs(beta=self.beta, loc=self.mu, scale=self.alpha, size=size)


class Rosenbrock(TargetDistribution):
    def __init__(self, n_var, n1, n2, mu=1, a=0.05, b=5):
        super().__init__(n_var)
        self.n1 = n1
        self.n2 = n2
        self.mu = mu
        self.a = a
        self.b = b

    def _get_parents(self, x):
        if self.n_var == 1:
            return np.array([np.sqrt(self.mu)])

        parents = np.zeros(self.n_var)

        # X_{1}
        parents[0] = np.sqrt(self.mu)

        # Default: each variable's parent is its previous
        parents[1:] = x[:-1]

        # X_{j, 2} special case for the start of each block
        for j in range(1, self.n2):
            parents[j * (self.n1 - 1) + 1] = x[0]

        return parents

    def logpi(self, x):
        if self.n_var == 1:
            return -self.a * (x - self.mu) ** 2

        parents = self._get_parents(x)

        return (-self.a * (x[0] - self.mu) ** 2) - (
            self.b * np.sum((x[1:] - parents[1:] ** 2) ** 2)
        )

    def d1_logpi(self, x):
        if self.n_var == 1:
            return -1 * self.a * (x - self.mu)

        d1_logpi = np.zeros(self.n_var)

        parents = self._get_parents(x)
        d = x - parents**2

        # X_{1}
        d1_logpi[0] = (-2 * self.a * d[0]) + (
            4 * self.b * x[0] * np.sum(d[1 :: self.n1 - 1])
        )

        # X_{j, i} for 2 <= i <= n1
        d1_logpi[1:] = -2 * self.b * d[1:]

        # X_{j, i} for 2 <= 1 < n1
        for j in range(self.n2):
            start = (self.n1 - 1) * j + 1
            stop = (self.n1 - 1) * (j + 1)
            d1_logpi[start:stop] += 4 * self.b * d[start + 1 : stop + 1] * x[start:stop]

        return d1_logpi

    def d2_logpi(self, x):
        if self.n_var == 1:
            return np.array([-2 * self.a])

        d2_logpi = np.zeros((self.n_var, self.n_var))

        parents = self._get_parents(x)
        d = x - (3 * parents**2)

        # 1. Diagonal entries
        # X_{1}
        d2_logpi[0, 0] = (-2 * self.a) + (4 * self.b * np.sum(d[1 :: self.n1 - 1]))

        # X_{j, i} 2 <= i <= n1
        diag_idx = np.diag_indices_from(d2_logpi)
        d2_logpi[diag_idx[0][1:], diag_idx[1][1:]] = -2 * self.b

        # X_{j, i} 2 <= i < n1
        for j in range(self.n2):
            start = (self.n1 - 1) * j + 1
            stop = (self.n1 - 1) * (j + 1)
            d2_logpi[diag_idx[0][start:stop], diag_idx[1][start:stop]] += (
                4 * self.b * d[start + 1 : stop + 1]
            )

        # 2. Off-diagonal entries
        for j in range(self.n2):
            for i in range(1, self.n1):
                idx = (self.n1 - 1) * j + i

                if i == 1:
                    d2_logpi[idx, 0] = 4 * self.b * x[0]
                    d2_logpi[0, idx] = d2_logpi[idx, 0]
                else:
                    d2_logpi[idx, idx - 1] = 4 * self.b * x[idx - 1]
                    d2_logpi[idx - 1, idx] = d2_logpi[idx, idx - 1]

        return d2_logpi

    def d3_logpi(self, x):
        return super().d3_logpi()

    def direct_sample(self, size):
        if self.n_var == 1:
            return rng.normal(loc=self.mu, scale=(1 / np.sqrt(2 * self.a)))

        n_dim = (self.n1 - 1) * self.n2 + 1

        X = np.empty((n_dim, size))

        for t in range(size):
            X[0, t] = rng.normal(lof=self.mu, scale=(1 / np.sqrt(2 * self.a)))

            for j in range(self.n2):
                for i in range(1, self.n1):
                    idx = (self.n1 - 1) * j + i

                    if i == 1:
                        x_parent = X[0, t]
                    else:
                        x_parent = X[idx - 1, t]

                    X[idx, t] = rng.normal(x_parent**2, 1 / np.sqrt(2 * self.b))

        return X

    def get_var_labels(self):
        labels = ["X_{1}"]

        for j in range(1, self.n2 + 1):
            for i in range(2, self.n1 + 1):
                labels.append(f"X_{{{j},{i}}}")

        return labels
