import numpy as np


class Rosenbrock():
    def __init__(self, n1, n2, mu=1, a=0.05, b=5):
        self.n1 = n1
        self.n2 = n2
        self.mu = mu
        self.a = a
        self.b = b


    def _get_parents(self, x):
        parents = np.zeros_like(x)

        # X_{1}
        parents[0] = np.sqrt(self.mu)

        # Default: each variable's parent is its previous
        parents[1:] = x[:-1]

        # X_{j, 2} special case for the start of each block
        for j in range(1, self.n2):
            parents[j * (self.n1-1) + 1] = x[0]

        return parents


    def logpi(self, x):
        parents = self._get_parents(x)

        return (- self.a * (x[0] - self.mu)**2) - (self.b * np.sum((x[1:] - parents[1:]**2)**2))


    def partial_logpi(self, x):
        partial_logpi = np.zeros_like(x)

        parents = self._get_parents(x)
        d = x - parents**2

        # X_{1}
        partial_logpi[0] = (- 2 * self.a * d[0]) + (4 * self.b * x[0] * np.sum(d[1::self.n1-1]))

        # X_{j, i} for 2 <= i <= n1
        partial_logpi[1:] = - 2 * self.b * d[1:]

        # X_{j, i} for 2 <= 1 < n1
        for j in range(self.n2):
            start = (self.n1 - 1) * j + 1
            stop = (self.n1 - 1) * (j + 1)
            partial_logpi[start:stop] += 4 * self.b * d[start+1:stop+1] * x[start:stop]
        
        return partial_logpi


    def hessian_logpi(self, x):
        hessian_logpi = np.zeros((x.shape[0], x.shape[0]))

        parents = self._get_parents(x)
        d = x - (3 * parents**2)

        # 1. Diagonal entries
        # X_{1}
        hessian_logpi[0, 0] = (- 2 * self.a) + (4 * self.b * np.sum(d[1::self.n1-1]))

        # X_{j, i} 2 <= i <= n1
        diag_idx = np.diag_indices_from(hessian_logpi)
        hessian_logpi[diag_idx[0][1:], diag_idx[1][1:]] = - 2 * self.b

        # X_{j, i} 2 <= i < n1
        for j in range (self.n2):
            start = (self.n1 - 1) * j + 1
            stop = (self.n1 - 1) * (j + 1)
            hessian_logpi[diag_idx[0][start:stop], diag_idx[1][start:stop]] += 4 * self.b * d[start+1:stop+1] 
        
        # 2. Off-diagonal entries
        for j in range(self.n2):
            for i in range(1, self.n1):
                idx = (self.n1-1) * j + i
                
                if i == 1:
                    hessian_logpi[idx, 0] = 4 * self.b * x[0]
                    hessian_logpi[0, idx] = hessian_logpi[idx, 0]
                else:
                    hessian_logpi[idx, idx-1] = 4 * self.b * x[idx-1]
                    hessian_logpi[idx-1, idx] = hessian_logpi[idx, idx-1]
        
        return hessian_logpi
    

    def sample(self, size):
        n_dim = (self.n1 - 1) * self.n2 + 1

        X = np.empty((n_dim, size))

        for t in range(size):
            X[0, t] = np.random.normal(self.mu, 1/np.sqrt(2*self.a))

            for j in range(self.n2):
                for i in range(1, self.n1):
                    idx = (self.n1-1) * j + i

                    if i == 1:
                        x_parent = X[0, t]
                    else:
                        x_parent = X[idx-1, t]

                    X[idx, t] = np.random.normal(x_parent**2, 1/np.sqrt(2*self.b))
        
        return X