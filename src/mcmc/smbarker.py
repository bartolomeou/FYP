import numpy as np
from src.utils.psd import project_to_psd


def SMBarker_proposal(x, partial_logpi_x, L_x, step_size, noise='normal'):
    if noise == 'bimodal':
        # z ~ Normal(mean = sigma, sd = 0.1 * sigma)
        z = np.random.normal(loc=step_size, scale=0.3*step_size, size=len(partial_logpi_x)) # TODO check
        
        # Acceptance probability for each component: 1 / (1 + exp(-grad * z))
        acceptance_prob = 1.0 / (1.0 + np.exp(- z * (partial_logpi_x @ L_x)))
        
        # b is either +1 or -1 depending on a uniform random draw
        b = 2 * (np.random.rand(len(partial_logpi_x)) < acceptance_prob) - 1

    else:
        # Magnitude
        z = step_size * np.random.normal(size=len(x), scale=1)

        # Direction
        threshold = 1 / (1 + np.exp(- z * (partial_logpi_x @ L_x)))
        b = np.where(np.random.uniform(size=1) < threshold, 1, -1) 

    return x + L_x @ (b * z)


def SMBarker_logq_ratio(x, y, partial_logpi_x, partial_logpi_y, L_x, L_y):
    z_xy = np.linalg.inv(L_x) @ (x - y)
    z_yx = np.linalg.inv(L_y) @ (y - x)

    logq_xy = - np.log1p(np.exp(z_xy @ (partial_logpi_x @ L_x)))
    logq_yx = - np.log1p(np.exp(z_yx @ (partial_logpi_y @ L_y)))

    return np.sum(logq_yx - logq_xy)


def SMBarker(target, n_iter, x_init, step_size=1, method='clip', noise='normal'):
    x = np.asarray(x_init)
    logpi_x = target.logpi(x)
    partial_logpi_x = target.partial_logpi(x)
    A_x = project_to_psd(np.linalg.inv(-target.hessian_logpi(x)), method)
    L_x = np.linalg.cholesky(A_x)

    # (#components, #iterations)
    X = np.empty((x.shape[0], n_iter))

    accepted = 0

    for i in range(n_iter):
        # Propose candidate state
        y = SMBarker_proposal(x, partial_logpi_x, L_x, noise, step_size)
        logpi_y = target.logpi(y)
        partial_logpi_y = target.partial_logpi(y)
        A_y = project_to_psd(np.linalg.inv(-target.hessian_logpi(y)), method)
        L_y = np.linalg.cholesky(A_y)

        log_acceptance = logpi_y - logpi_x + SMBarker_logq_ratio(x, y, partial_logpi_x, partial_logpi_y, L_x, L_y)

        # Acceptance criterion
        if np.log(np.random.uniform(size=1)) < log_acceptance:
            x = y
            logpi_x = logpi_y
            partial_logpi_x = partial_logpi_y
            L_x = L_y

            accepted += 1

        X[:, i] = x
        
    acceptance_rate = accepted / n_iter

    return X, acceptance_rate
