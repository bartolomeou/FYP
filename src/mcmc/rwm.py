import numpy as np


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