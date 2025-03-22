import numpy as np


def softabs(lamb, alpha=1e6, small=1e-6, large=10):
    x = alpha * lamb

    # Near zero: lambda*coth(alpha*lambda) ~ 1/alpha + (alpha*lambda^2)/3 + ...
    if abs(x) < small:
        return 1.0 / alpha

    # Large positive: coth(x) ~ 1, so lambda*coth(x) ~ lambda
    # Large negative: coth(x) ~ -1, so lambda*coth(x) ~ -lambda
    if abs(x) > large:
        return abs(lamb)

    # Mid-range: directly compute lambda*coth(alpha*lambda) = lambda*cosh(x)/sinh(x)
    # return lamb * (np.cosh(x) / np.sinh(x))
    return lamb / np.tanh(x)


def project_to_pd(A, method="clip", epsilon=1e-8, alpha=1.0):
    """
    Project a matrix to the positive definite cone.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    if method == "clip":
        eigenvalues = np.maximum(eigenvalues, epsilon)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    elif method == "abs":
        eigenvalues = np.abs(eigenvalues)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    elif method == "softabs":
        eigenvalues_soft = np.array(
            [softabs(eigenvalue, alpha) for eigenvalue in eigenvalues]
        )
        return eigenvectors @ np.diag(eigenvalues_soft) @ eigenvectors.T

    else:
        return A


def is_pd(A):
    A = (A + A.T) / 2

    eigenvalues = np.linalg.eigvalsh(A)

    # Check if all eigenvalues are non-negative (within tolerance)
    return np.all(eigenvalues > 0)
