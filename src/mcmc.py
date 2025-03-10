from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.special import logsumexp

from src.adapter import adapter
from src.utils.pd import project_to_pd


class MetropolisHastingsMCMC(ABC):
    def __init__(self, target_accept_prob=None):
        self.target_accept_prob = target_accept_prob

        self.target = None
        self.n_var = None

        self.h = None

        self.x = None
        self.logpi_x = None

        self.y = None
        self.logpi_y = None

        self.rng = None

    @abstractmethod
    def initialize(self, initial_state):
        self.x = np.atleast_1d(initial_state)
        self.logpi_x = self.target.logpi(self.x)

        self.n_var = self.x.shape[0]

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

        accepted = 0
        if np.log(self.rng.uniform()) < log_accept_prob:
            self.update()
            accepted = 1

        return {
            "accept_prob": np.exp(min(0, log_accept_prob)),
            "accepted": accepted,
        }

    def sample(
        self,
        target,
        initial_state=None,
        n_main_iter=5000,
        step_size=1,
        n_burnin_iter=5000,
        adapter_method="default",
        lr=0.1,
        batch_size=30,
        seed=2025,
    ):
        self.rng = np.random.default_rng(seed)

        self.target = target
        self.h = step_size

        if initial_state is None:
            initial_state = np.zeros(target.n_var)
        self.initialize(initial_state)

        X_burnin = np.empty((self.n_var, n_burnin_iter))
        accept_prob_burnin = []

        for i in range(n_burnin_iter):
            accept_prob = self.step()["accept_prob"]
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
        accept_count = 0

        for i in range(n_main_iter):
            accept_count += self.step()["accepted"]

            X_main[:, i] = self.x

        return {
            "trace_main": X_main,
            "trace_burnin": X_burnin,
            "accept_rate": accept_count / n_main_iter,
        }


class RandomWalk(MetropolisHastingsMCMC):
    def __init__(self, target_accept_prob=0.234):
        super().__init__(target_accept_prob)

    def initialize(self, initial_state):
        super().initialize(initial_state)

    def propose(self):
        return self.x + self.h * self.rng.normal(size=self.n_var)

    def log_q_ratio(self):
        return 0

    def update(self):
        super().update()


class Barker(MetropolisHastingsMCMC):
    def __init__(self, target_accept_prob=0.574, noise="normal"):
        super().__init__(target_accept_prob)
        self.noise = noise

        self.d1_logpi_x = None
        self.d1_logpi_y = None

    def initialize(self, initial_state):
        super().initialize(initial_state)
        self.d1_logpi_x = self.target.d1_logpi(self.x)

    def propose(self):
        if self.noise == "bimodal":
            # z ~ Normal(mean = sigma, sd = a * sigma) for some constant a
            z = self.rng.normal(loc=self.h, scale=0.3 * self.h, size=self.n_var)
        else:
            # z ~ Normal(mean = 0, sd = sigma)
            z = self.rng.normal(loc=0, scale=self.h, size=self.n_var)

        # Acceptance probability for each component: 1 / (1 + exp(-grad * z))
        p_xz = 1 / (1 + np.exp(-z * self.d1_logpi_x))

        # b is either +1 or -1 depending on a uniform random draw
        b = 2 * (self.rng.uniform(size=self.n_var) < p_xz) - 1

        return self.x + b * z

    def log_q_ratio(self):
        self.d1_logpi_y = self.target.d1_logpi(self.y)

        z = self.y - self.x

        logq_xy = -logsumexp(
            [np.zeros_like(z), -z * self.d1_logpi_x], axis=0
        )  # -np.log1p(np.exp(-z * self.d1_logpi_x))
        logq_yx = -logsumexp(
            [np.zeros_like(z), z * self.d1_logpi_y], axis=0
        )  # -np.log1p(np.exp(z * self.d1_logpi_y))

        return np.sum(logq_yx - logq_xy)

    def update(self):
        super().update()
        self.d1_logpi_x = self.d1_logpi_y


class SMBarker(MetropolisHastingsMCMC):
    def __init__(
        self,
        target_accept_prob=0.574,
        noise="normal",
        psd_method="clip",
    ):
        super().__init__(target_accept_prob)
        self.noise = noise
        self.psd_method = psd_method

        self.d1_logpi_x = None
        self.L_x = None

        self.d1_logpi_y = None
        self.L_y = None

    def initialize(self, initial_state):
        super().initialize(initial_state)
        self.d1_logpi_x = self.target.d1_logpi(self.x)

        if self.n_var == 1:
            self.L_x = np.sqrt(-1 / self.target.d2_logpi(self.x))
        else:
            A_x = project_to_pd(
                np.linalg.inv(-self.target.d2_logpi(self.x)), method=self.psd_method
            )
            self.L_x = np.linalg.cholesky(A_x)  # Lower-triangular Cholesky factor

    def propose(self):
        if self.noise == "bimodal":
            # z ~ Normal(mean = sigma, sd = a * sigma) for some constant a
            z = self.rng.normal(loc=self.h, scale=0.3 * self.h, size=self.n_var)
        else:
            # z ~ Normal(mean = 0, sd = sigma)
            z = self.rng.normal(loc=0, scale=self.h, size=self.n_var)

        p_xz = 1 / (1 + np.exp(-z * (self.d1_logpi_x @ self.L_x)))

        b = 2 * (self.rng.uniform(size=self.n_var) < p_xz) - 1

        # @ still works in one-dimension case because L_x, b, z are 1-D array
        return self.x + self.L_x @ (b * z)

    def log_q_ratio(self):
        self.d1_logpi_y = self.target.d1_logpi(self.y)

        if self.n_var == 1:
            self.L_y = np.sqrt(-1 / self.target.d2_logpi(self.y))

            z_xy = (1 / self.L_x) * (self.x - self.y)
            z_yx = (1 / self.L_y) * (self.y - self.x)
        else:
            A_y = project_to_pd(
                np.linalg.inv(-self.target.d2_logpi(self.y)), method=self.psd_method
            )
            self.L_y = np.linalg.cholesky(A_y)

            z_xy = np.linalg.inv(self.L_x) @ (self.x - self.y)
            z_yx = np.linalg.inv(self.L_y) @ (self.y - self.x)

        # Changed @ to *, might explain the weird bimodal behaviour
        logq_xy = -logsumexp(
            [np.zeros_like(z_xy), z_xy * (self.d1_logpi_x @ self.L_x)], axis=0
        )  # -np.log1p(np.exp(z_xy * (self.d1_logpi_x @ self.L_x)))
        logq_yx = logsumexp(
            [np.zeros_like(z_yx), z_yx * (self.d1_logpi_y @ self.L_y)], axis=0
        )  # -np.log1p(np.exp(z_yx * (self.d1_logpi_y @ self.L_y)))

        return np.sum(logq_yx - logq_xy)

    def update(self):
        super().update()
        self.d1_logpi_x = self.d1_logpi_y
        self.L_x = self.L_y


class MALA(MetropolisHastingsMCMC):
    def __init__(self, target_accept_prob=0.574):
        super().__init__(target_accept_prob)
        self.d1_logpi_x = None
        self.d1_logpi_y = None

    def initialize(self, initial_state):
        super().initialize(initial_state)
        self.d1_logpi_x = self.target.d1_logpi(self.x)

    def propose(self):
        z = self.rng.normal(loc=0, scale=self.h, size=self.n_var)
        return self.x + (1 / 2) * (self.h**2) * self.d1_logpi_x + z

    def log_q_ratio(self):
        self.d1_logpi_y = self.target.d1_logpi(self.y)

        if self.n_var == 1:
            log_xy = norm.logpdf(
                self.y,
                loc=(self.x + (1 / 2) * (self.h**2) * self.d1_logpi_x),
                scale=self.h,
            )
            log_yx = norm.logpdf(
                self.x,
                loc=(self.y + (1 / 2) * (self.h**2) * self.d1_logpi_y),
                scale=self.h,
            )
        else:
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
    def __init__(self, target_accept_prob=0.574, psd_method="abs"):
        super().__init__(target_accept_prob)
        self.psd_method = psd_method

        self.d1_logpi_x = None
        self.A_x = None
        self.L_x = None

        self.d1_logpi_y = None
        self.A_y = None
        self.L_y = None

    def initialize(self, initial_state):
        super().initialize(initial_state)
        self.d1_logpi_x = self.target.d1_logpi(self.x)

        if self.n_var == 1:
            self.A_x = 1 / (-self.target.d2_logpi(self.x))
            self.L_x = np.sqrt(self.A_x)
        else:
            self.A_x = np.linalg.inv(-self.target.d2_logpi(self.x))
            self.L_x = np.linalg.cholesky(
                project_to_pd(self.A_x, method=self.psd_method)
            )

    def propose(self):
        z = self.rng.normal(loc=0, scale=self.h, size=self.n_var)

        if self.L_x is None:  # L_x is always not None for 1d
            self.L_x = np.linalg.cholesky(
                project_to_pd(self.A_x, method=self.psd_method)
            )

        if self.n_var == 1:
            return (
                self.x
                + (1 / 2) * (self.h**2) * self.A_x * self.d1_logpi_x
                + self.L_x * z
            )
        else:
            return (
                self.x
                + (1 / 2) * (self.h**2) * self.A_x @ self.d1_logpi_x
                + self.L_x @ z
            )

    def log_q_ratio(self):
        self.d1_logpi_y = self.target.d1_logpi(self.y)

        if self.n_var == 1:
            self.A_y = 1 / (-self.target.d2_logpi(self.y))
            self.L_y = np.sqrt(self.A_y)

            log_xy = norm.logpdf(
                self.y,
                loc=(self.x + (1 / 2) * (self.h**2) * self.A_x * self.d1_logpi_x),
                scale=(self.h * self.L_x),
            )
            log_yx = norm.logpdf(
                self.x,
                loc=(self.y + (1 / 2) * (self.h**2) * self.A_y * self.d1_logpi_y),
                scale=(self.h * self.L_y),
            )
        else:
            self.A_y = np.linalg.inv(-self.target.d2_logpi(self.y))

            log_xy = multivariate_normal.logpdf(
                self.y,
                mean=self.x + (1 / 2) * (self.h**2) * self.A_x @ self.d1_logpi_x,
                cov=project_to_pd((self.h**2) * self.A_x, method=self.psd_method),
            )
            log_yx = multivariate_normal.logpdf(
                self.x,
                mean=self.y + (1 / 2) * (self.h**2) * self.A_y @ self.d1_logpi_y,
                cov=project_to_pd((self.h**2) * self.A_y, method=self.psd_method),
            )

        return log_yx - log_xy

    def update(self):
        super().update()
        self.d1_logpi_x = self.d1_logpi_y
        self.A_x = self.A_y
        self.L_x = self.L_y


class MMALA(MetropolisHastingsMCMC):
    # Only for GND
    def __init__(self, target_accept_prob=0.574):
        super().__init__(target_accept_prob)
        self.d1_logpi_x = None
        self.A_x = None
        self.L_x = None
        self.Gamma_x = None

        self.d1_logpi_y = None
        self.A_y = None
        self.L_y = None
        self.Gamma_y = None

    def initialize(self, initial_state):
        super().initialize(initial_state)
        self.d1_logpi_x = self.target.d1_logpi(self.x)

        d2_logpi_x = self.target.d2_logpi(self.x)
        self.A_x = 1 / (-d2_logpi_x)
        self.L_x = np.sqrt(self.A_x)
        self.Gamma_x = (
            -(1 / 2)
            * (self.A_x**2)
            * np.sign(d2_logpi_x)
            * self.target.d3_logpi(self.x)
        )

    def propose(self):
        z = self.rng.normal(loc=0, scale=self.h, size=self.n_var)

        return (
            self.x
            + (1 / 2) * (self.h**2) * self.A_x * self.d1_logpi_x
            + (self.h**2) * self.Gamma_x
            + self.L_x * z
        )

    def log_q_ratio(self):
        self.d1_logpi_y = self.target.d1_logpi(self.y)

        d2_logpi_y = self.target.d2_logpi(self.y)
        self.A_y = 1 / (-d2_logpi_y)
        self.L_y = np.sqrt(self.A_y)
        self.Gamma_y = (
            -(1 / 2)
            * (self.A_y**2)
            * np.sign(d2_logpi_y)
            * self.target.d3_logpi(self.y)
        )

        log_xy = norm.logpdf(
            self.y,
            loc=self.x
            + (1 / 2) * (self.h**2) * self.A_x * self.d1_logpi_x
            + (self.h**2) * self.Gamma_x,
            scale=self.h * self.L_x,
        )
        log_yx = norm.logpdf(
            self.x,
            loc=self.y
            + (1 / 2) * (self.h**2) * self.A_y * self.d1_logpi_y
            + (self.h**2) * self.Gamma_y,
            scale=self.h * self.L_y,
        )

        return log_yx - log_xy

    def update(self):
        super().update()
        self.d1_logpi_x = self.d1_logpi_y
        self.A_x = self.A_y
        self.L_x = self.L_y
        self.Gamma_x = self.Gamma_y
