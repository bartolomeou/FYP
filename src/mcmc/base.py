from abc import ABC, abstractmethod
import numpy as np


class MetropolisHastingsMCMC(ABC):
    def __init__(self, target_accept_prob=None, var_labels=None):
        self.h = None
        self.target_accept_prob = target_accept_prob

        self.x = None
        self.logpi_x = None

        self.y = None
        self.logpi_y = None

        self.n_var = None
        self.var_labels = var_labels

    @abstractmethod
    def log_q_ratio(self):
        pass

    @abstractmethod
    def propose(self):
        pass

    @abstractmethod
    def update(self):
        self.x = self.y
        self.logpi_x = self.logpi_y

    def sample(
        self, target, initial_state, n_main_iter, n_burnin_iter, adapter, step_size=1
    ):
        self.h = step_size

        self.x = np.asarray(initial_state)
        self.logpi_x = target.logpi(self.x)

        self.n_var = self.x.shape[0]

        X_burnin = np.empty((self.n_var, n_burnin_iter))
        accept_prob = []

        for i in range(n_burnin_iter):
            self.y = self.propose()
            self.logpi_y = target.logpi(self.y)

            log_accept_rate = self.logpi_x - self.logpi_y + self.log_q_ratio()

            if np.log(np.random.uniform()) < log_accept_rate:
                self.update()

            X_burnin[:, i] = self.x

            accept_prob.append(min(1, np.exp(log_accept_rate)))
            self.h = adapter(self.h, accept_prob, self.target_accept_prob)

        X_main = np.empty((self.n_var, n_main_iter))
        accept_count = 0

        for i in range(n_main_iter):
            self.y = self.propose()
            self.logpi_y = target.logpi(self.y)

            log_accept_rate = self.logpi_y - self.logpi_x + self.log_q_ratio()

            if np.log(np.random.uniform()) < log_accept_rate:
                self.update()
                accept_count += 1

            X_main[:, i] = self.x

        return {
            "trace_burnin": X_burnin,
            "trace_main": X_main,
            "accept_rate": accept_count / n_main_iter,
        }
