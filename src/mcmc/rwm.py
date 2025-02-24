import numpy as np
from .base import MetropolisHastingsMCMC


class RandomWalk(MetropolisHastingsMCMC):
    def __init__(self, target_accept_prob=0.234, var_labels=None):
        super().__init__(target_accept_prob, var_labels)

    def log_q_ratio(self):
        return 0

    def propose(self):
        return self.x + self.h * np.random.normal(size=self.n_var)

    def update(self):
        super().update()


def RWM_proposal(x, step_size):
    return x + step_size * np.random.normal(size=x.shape[0])


def RWM(target, n_iter, x_init, step_size=1):
    x = np.asarray(x_init)
    logpi_x = target.logpi(x)

    # (#components, #iterations)
    X = np.empty((x.shape[0], n_iter))

    accepted = 0

    for i in range(n_iter):
        # Proposal state
        y = RWM_proposal(x, step_size)
        logpi_y = target.logpi(y)

        log_acceptance = logpi_y - logpi_x

        # Acceptance criterion
        if np.log(np.random.uniform(size=1)) < log_acceptance:
            x = y
            logpi_x = logpi_y
            accepted += 1

        X[:, i] = x

    acceptance_rate = accepted / n_iter

    return X, acceptance_rate
