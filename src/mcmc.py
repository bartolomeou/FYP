from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

from src.adapter import adapter
from src.utils.psd import project_to_psd

rng = np.random.default_rng()


class MetropolisHastingsMCMC(ABC):
    def __init__(self, target_accept_prob=None, var_labels=None):
        self.target_accept_prob = target_accept_prob

        self.h = None
        self.target = None
        self.n_var = None
        self.var_labels = var_labels

        self.x = None
        self.logpi_x = None

        self.y = None
        self.logpi_y = None

    @abstractmethod
    def propose(self):
        pass

    @abstractmethod
    def log_q_ratio(self):
        pass

    @abstractmethod
    def update(self):
        self.x = self.y
        self.logpi_x = self.logpi_y

    def step(self):
        self.y = self.propose()
        self.logpi_y = self.target.logpi(self.y)

        log_accept_prob = self.logpi_y - self.logpi_x + self.log_q_ratio()

        if np.log(rng.uniform()) < log_accept_prob:
            self.update()

        return np.exp(min(0, log_accept_prob))

    def sample(
        self,
        target,
        initial_state,
        n_main_iter,
        step_size=1,
        n_burnin_iter=0,
        adapter_method="default",
        lr=0.1,
        batch_size=30,
    ):
        self.target = target
        self.h = step_size

        self.x = np.asarray(initial_state)
        self.logpi_x = target.logpi(self.x)

        self.n_var = self.x.shape[0]

        X_burnin = np.empty((self.n_var, n_burnin_iter))
        accept_prob_burnin = []

        for i in range(n_burnin_iter):
            accept_prob = self.step()
            accept_prob_burnin.append(accept_prob)

            self.h = adapter(
                self.h,
                accept_prob_burnin,
                self.target_accept_prob,
                adapter_method,
                lr,
                batch_size,
            )

            X_burnin[:, i] = self.x

        X_main = np.empty((self.n_var, n_main_iter))
        accept_prob_main = []

        for i in range(n_main_iter):
            accept_prob = self.step()
            accept_prob_main.append(accept_prob)

            X_main[:, i] = self.x

        return {
            "trace_main": X_main,
            "accept_prob_main": accept_prob_main,
            "trace_burnin": X_burnin,
            "accept_prob_burnin": accept_prob_burnin,
        }


class RandomWalk(MetropolisHastingsMCMC):
    def __init__(self, target_accept_prob=0.234, var_labels=None):
        super().__init__(target_accept_prob, var_labels)

    def propose(self):
        return self.x + self.h * rng.normal(size=self.n_var)

    def log_q_ratio(self):
        return 0

    def update(self):
        super().update()


class Barker(MetropolisHastingsMCMC):
    def __init__(self, target_accept_prob=0.574, var_labels=None, noise="normal"):
        super().__init__(target_accept_prob, var_labels)
        self.noise = noise

        self.d1_logpi_x = None
        self.d1_logpi_y = None

    def propose(self):
        if self.d1_logpi_x is None:
            self.d1_logpi_x = self.target.d1_logpi(self.x)

        if self.noise == "bimodal":
            # z ~ Normal(mean = sigma, sd = a * sigma) for some constant a
            z = rng.normal(loc=self.h, scale=0.3 * self.h, size=self.n_var)
        else:
            # z ~ Normal(mean = 0, sd = sigma)
            z = rng.normal(loc=0, scale=self.h, size=self.n_var)

        # Acceptance probability for each component: 1 / (1 + exp(-grad * z))
        threshold = 1 / (1 + np.exp(-z * self.d1_logpi_x))

        # b is either +1 or -1 depending on a uniform random draw
        b = 2 * (rng.uniform(size=self.n_var) < threshold) - 1

        return self.x + b * z

    def log_q_ratio(self):
        self.d1_logpi_y = self.target.d1_logpi(self.y)

        z = self.y - self.x

        logq_xy = -logsumexp(
            np.stack([np.zeros_like(z), -z * self.d1_logpi_x]), axis=0
        )  # -np.log1p(np.exp(-z * self.d1_logpi_x))
        logq_yx = -logsumexp(
            np.stack([np.zeros_like(z), z * self.d1_logpi_y]), axis=0
        )  # -np.log1p(np.exp(z * self.d1_logpi_y))

        return np.sum(logq_yx - logq_xy)

    def update(self):
        super().update()
        self.d1_logpi_x = self.d1_logpi_y


class SMBarker(MetropolisHastingsMCMC):
    def __init__(
        self,
        target_accept_prob=0.574,
        var_labels=None,
        noise="normal",
        psd_method="clip",
    ):
        super().__init__(target_accept_prob, var_labels)
        self.noise = noise
        self.psd_method = psd_method

        self.d1_logpi_x = None
        self.L_x = None

        self.d1_logpi_y = None
        self.L_y = None

    def propose(self):
        if self.d1_logpi_x is None:
            self.d1_logpi_x = self.target.d1_logpi(self.x)
            A_x = project_to_psd(
                np.linalg.inv(-self.target.d2_logpi(self.x)), method=self.psd_method
            )
            self.L_x = np.linalg.cholesky(A_x)  # Lower-triangular Cholesky factor

        if self.noise == "bimodal":
            # z ~ Normal(mean = sigma, sd = a * sigma) for some constant a
            z = rng.normal(loc=self.h, scale=0.3 * self.h, size=self.n_var)
        else:
            # z ~ Normal(mean = 0, sd = sigma)
            z = rng.normal(loc=0, scale=self.h, size=self.n_var)

        threshold = 1 / (1 + np.exp(-z * (self.d1_logpi_x @ self.L_x)))

        b = 2 * (rng.uniform(size=self.n_var) < threshold) - 1

        return self.x + self.L_x @ (b * z)

    def log_q_ratio(self):
        self.d1_logpi_y = self.target.d1_logpi(self.y)
        A_y = project_to_psd(
            np.linalg.inv(-self.target.d2_logpi(self.y)), method=self.psd_method
        )
        self.L_y = np.linalg.cholesky(A_y)

        z_xy = np.linalg.inv(self.L_x) @ (self.x - self.y)
        z_yx = np.linalg.inv(self.L_y) @ (self.y - self.x)

        # NOTE: Changed @ to *, might explain the weird bimodal behaviour
        logq_xy = -logsumexp(
            np.stack([np.zeros_like(z_xy), z_xy * (self.d1_logpi_x @ self.L_x)]), axis=0
        )  # -np.log1p(np.exp(z_xy * (self.d1_logpi_x @ self.L_x)))
        logq_yx = logsumexp(
            np.stack([np.zeros_like(z_yx), z_yx * (self.d1_logpi_y @ self.L_y)]), axis=0
        )  # -np.log1p(np.exp(z_yx * (self.d1_logpi_y @ self.L_y)))

        return np.sum(logq_yx - logq_xy)

    def update(self):
        super().update()
        self.d1_logpi_x = self.d1_logpi_y
        self.L_x = self.L_y


class MALA(MetropolisHastingsMCMC):
    def __init__(self, target_accept_prob=0.574, var_labels=None):
        super().__init__(target_accept_prob, var_labels)
        self.d1_logpi_x = None
        self.d1_logpi_y = None

    def propose(self):
        if self.d1_logpi_x is None:
            self.d1_logpi_x = self.target.d1_logpi(self.x)

        z = rng.normal(loc=0, scale=self.h, size=self.n_var)
        return self.x + (1 / 2) * (self.h**2) * self.d1_logpi_x + z

    def log_q_ratio(self):
        self.d1_logpi_y = self.target.d1_logpi(self.y)

        log_xy = multivariate_normal.logpdf(
            self.y,
            mean=(self.x + (1 / 2) * (self.h**2) * self.d1_logpi_x),
            cov=(self.h**2),
        )
        log_yx = multivariate_normal.logpdf(
            self.x,
            mean=(self.y + (1 / 2) * (self.h**2) * self.d1_logpi_y),
            cov=(self.h**2),
        )

        return log_yx - log_xy

    def update(self):
        super().update()
        self.d1_logpi_x = self.d1_logpi_y


class SMMALA(MetropolisHastingsMCMC):
    def __init__(self, target_accept_prob=0.574, var_labels=None, psd_method="abs"):
        super().__init__(target_accept_prob, var_labels)
        self.psd_method = psd_method

        self.d1_logpi_x = None
        self.A_x = None

        self.d1_logpi_y = None
        self.A_y = None

    def propose(self):
        if self.d1_logpi_x is None:
            self.d1_logpi_x = self.target.d1_logpi(self.x)
            self.A_x = np.linalg.inv(-self.target.d2_logpi(self.x))

        z = rng.normal(loc=0, scale=self.h, size=self.n_var)

        L_x = np.linalg.cholesky(project_to_psd(self.A_x, method=self.psd_method))

        return self.x + (1 / 2) * (self.h**2) * self.A_x @ self.d1_logpi_x + L_x @ z

    def log_q_ratio(self):
        self.d1_logpi_y = self.target.d1_logpi(self.y)
        self.A_y = np.linalg.inv(-self.target.d2_logpi(self.y))

        mean_xy = self.x + (1 / 2) * (self.h**2) * self.A_x @ self.d1_logpi_x
        mean_yx = self.y + (1 / 2) * (self.h**2) * self.A_y @ self.d1_logpi_y

        cov_xy = (self.h**2) * self.A_x
        cov_yx = (self.h**2) * self.A_y

        log_xy = multivariate_normal.logpdf(
            self.y, mean=mean_xy, cov=project_to_psd(cov_xy, method=self.psd_method)
        )
        log_yx = multivariate_normal.logpdf(
            self.x, mean=mean_yx, cov=project_to_psd(cov_yx, method=self.psd_method)
        )

        return log_yx - log_xy

    def update(self):
        super().update()
        self.d1_logpi_x = self.d1_logpi_y
        self.A_x = self.A_y
