import numpy as np
from scipy.stats import multivariate_normal


def MALA_proposal(x, grad_logpi_x, step_size):
    z = step_size * np.random.normal(size=x.shape[0])
    return x + (1 / 2) * (step_size**2) * grad_logpi_x + z


def MALA_logq_ratio(x, y, grad_logpi_x, grad_logpi_y, step_size):
    log_xy = multivariate_normal.logpdf(
        y, mean=(x + (1 / 2) * (step_size**2) * grad_logpi_x), cov=(step_size**2)
    )
    log_yx = multivariate_normal.logpdf(
        x, mean=(y + (1 / 2) * (step_size**2) * grad_logpi_y), cov=(step_size**2)
    )

    return log_yx - log_xy


def MALA(target, n_iter, x_init, step_size=1):
    x = np.asarray(x_init)
    logpi_x = target.logpi(x)
    grad_logpi_x = target.d1_logpi(x)

    # (#components, #iterations)
    X = np.empty((x.shape[0], n_iter))

    accepted = 0

    for i in range(n_iter):
        # Proposal state
        y = MALA_proposal(x, grad_logpi_x, step_size)
        logpi_y = target.logpi(y)
        grad_logpi_y = target.d1_logpi(y)

        log_acceptance = (
            logpi_y
            - logpi_x
            + MALA_logq_ratio(x, y, grad_logpi_x, grad_logpi_y, step_size)
        )

        # Acceptance criterion
        if np.log(np.random.uniform(size=1)) < log_acceptance:
            x = y
            logpi_x = logpi_y
            grad_logpi_x = grad_logpi_y

            accepted += 1

        X[:, i] = x

    acceptance_rate = accepted / n_iter

    return X, acceptance_rate
