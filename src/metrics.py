import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import arviz as az


def ks_distance(X_mcmc, target, sample_size=5000, verbose=False):
    labels = target.get_var_labels()

    X_true = target.direct_sample(sample_size)

    ks, _ = ks_2samp(X_mcmc, X_true, axis=1)

    if verbose:
        print("Kolmogorov-Smirnov Distance (KS)")

        for i in range(len(ks)):
            print("\t", labels[i], ": ", ks[i])

    return ks


def ess(X_mcmc, target, verbose=False):
    labels = target.get_var_labels()

    ess = []

    for i in range(target.n_var):
        ess.append(az.ess(X_mcmc[i, :]))

    if verbose:
        print("Effective Sample Size (ESS)")

        for i in range(len(ess)):
            print("\t", labels[i], ": ", ess[i])

    return ess


def quantiles(
    X_mcmc, target, quantiles=[0.025, 0.25, 0.5, 0.75, 0.975], sample_size=5000
):
    labels = target.get_var_labels()

    X_true = target.direct_sample(sample_size)

    df = pd.DataFrame(columns=[str(q) for q in quantiles])

    for i in range(target.n_var):
        df.loc["True" + labels[i]] = np.quantile(X_true[i, :], quantiles)
        df.loc["MCMC" + labels[i]] = np.quantile(X_true[i, :], quantiles)

    return df
