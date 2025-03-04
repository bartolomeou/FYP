import numpy as np


def softabs(lam, alpha=1.0, small=1e-8, large=20.0):
    x = alpha * lam

    # Near zero: lam*coth(alpha*lam) ~ 1/alpha + alpha*lam^2/3 + ...
    if abs(x) < small:
        return 1.0 / alpha

    # Large positive: coth(x) ~ 1, so lam*coth(x) ~ lam
    # Large negative: coth(x) ~ -1, so lam*coth(x)
    if abs(x) > large:
        return abs(lam)

    # Otherwise, directly compute lam*coth(alpha*lam) = lam*cosh(x)/sinh(x)
    return lam * (np.cosh(x) / np.sinh(x))


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
        return (
            eigenvectors
            @ np.diag(np.maximum(eigenvalues_soft, epsilon))
            @ eigenvectors.T
        )

    else:
        return A


def is_pd(A):
    A = (A + A.T) / 2

    eigenvalues = np.linalg.eigvalsh(A)

    # Check if all eigenvalues are non-negative (within tolerance)
    return np.all(eigenvalues > 0)
