import numpy as np
from scipy.stats import multivariate_normal
from src.utils.psd import project_to_psd


def SMMALA_proposal(x, grad_logpi_x, A_x, step_size, method):
    z = step_size * np.random.normal(size=x.shape[0]) 

    L_x = np.linalg.cholesky(project_to_psd(A_x, method))

    return x + (1/2) * (step_size**2) * A_x @ grad_logpi_x + L_x @ z


def SMMALA_logq_ratio(x, y, grad_logpi_x, grad_logpi_y, A_x, A_y, step_size, method):
    mean_xy = x + (1/2) * (step_size**2) * A_x @ grad_logpi_x
    mean_yx = y + (1/2) * (step_size**2) * A_y @ grad_logpi_y

    cov_xy = (step_size**2) * A_x
    cov_yx = (step_size**2) * A_y

    log_xy = multivariate_normal.logpdf(y, mean=mean_xy, cov=project_to_psd(cov_xy, method))
    log_yx = multivariate_normal.logpdf(x, mean=mean_yx, cov=project_to_psd(cov_yx, method))
    
    return log_yx - log_xy


def SMMALA(target, n_iter, x_init, step_size=1, method='clip'):
    x = np.asarray(x_init)
    logpi_x = target.logpi(x)    
    grad_logpi_x = target.d1_logpi(x)
    A_x = np.linalg.inv(-target.d2_logpi(x))

    # (#components, #iterations)
    X = np.empty((x.shape[0], n_iter))

    accepted = 0

    for i in range(n_iter):
        # Proposal state
        y = SMMALA_proposal(x, grad_logpi_x, A_x, step_size, method)
        logpi_y = target.logpi(y)
        grad_logpi_y = target.d1_logpi(y)
        A_y = np.linalg.inv(-target.d2_logpi(y))

        log_acceptance = logpi_y - logpi_x + SMMALA_logq_ratio(x, y, grad_logpi_x, grad_logpi_y, A_x, A_y, step_size, method)

        # Acceptance criterion
        if np.log(np.random.uniform(size=1)) < log_acceptance:
            x = y
            logpi_x = logpi_y
            grad_logpi_x = grad_logpi_y
            A_x = A_y

            accepted += 1

        X[:, i] = x
        
    acceptance_rate = accepted / n_iter

    return X, acceptance_rate