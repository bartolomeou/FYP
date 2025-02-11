import numpy as np


def Barker_proposal(x, grad_logpi_x, step_size, noise='normal'):
    if noise == 'bimodal':
        # z ~ Normal(mean = sigma, sd = 0.1 * sigma)
        z = np.random.normal(loc=step_size, scale=0.3*step_size, size=len(grad_logpi_x))
        
        # Acceptance probability for each component: 1 / (1 + exp(-grad * z))
        acceptance_prob = 1.0 / (1.0 + np.exp(-grad_logpi_x * z))
        
        # b is either +1 or -1 depending on a uniform random draw
        b = 2 * (np.random.rand(len(grad_logpi_x)) < acceptance_prob) - 1

    else:
        # Magnitude
        z = step_size * np.random.normal(size=len(x), scale=1)

        # Direction
        threshold = 1 / (1 + np.exp(- z * grad_logpi_x))
        b = np.where(np.random.uniform(size=1) < threshold, 1, -1) 
    
    return x + b * z


def Barker_logq_ratio(x, y, grad_logpi_x, grad_logpi_y):
    z = y - x

    logq_xy = - np.log1p(np.exp(- z * grad_logpi_x))
    logq_yx = - np.log1p(np.exp(z * grad_logpi_y))

    return np.sum(logq_yx - logq_xy)
   

def Barker(target, n_iter, x_init, step_size=1, noise='normal'):
    x = np.asarray(x_init)
    logpi_x = target.logpi(x)
    grad_logpi_x = target.d1_logpi(x)

    # (#components, #iterations)
    X = np.empty((x.shape[0], n_iter))

    accepted = 0
    
    for i in range(n_iter):
        # Proposal state
        y = Barker_proposal(x, grad_logpi_x, step_size, noise)
        logpi_y = target.logpi(y)
        grad_logpi_y = target.d1_logpi(y)

        log_acceptance = logpi_y - logpi_x + Barker_logq_ratio(x, y, grad_logpi_x, grad_logpi_y)
        
        # Acceptance criterion
        if np.log(np.random.uniform(size=1)) < log_acceptance:
            x = y
            logpi_x = logpi_y
            grad_logpi_x = grad_logpi_y

            accepted +=1

        X[:, i] = x
    
    acceptance_rate = accepted / n_iter

    return X, acceptance_rate 