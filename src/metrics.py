import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from scipy.stats import anderson_ksamp
import arviz as az


def ks_distance(X_mcmc, target, X_true=None, sample_size=10000, verbose=False):
    if X_true is None:
        X_true = target.direct_sample(sample_size)

    if target.n_var == 1:
        X_true = X_true.reshape(-1, 1)

    ks = ks_2samp(X_mcmc, X_true, axis=1).statistic

    if verbose:
        labels = target.get_var_labels()

        print("Kolmogorov-Smirnov Distance (KS)")

        for i in range(target.n_var):
            print("\t", labels[i], ": ", ks[i])

    return ks


def ad_distance(X_mcmc, target, X_true=None, sample_size=10000, verbose=False):
    if X_true is None:
        X_true = np.atleast_1d(target.direct_sample(sample_size))

    if target.n_var == 1:
        X_true = X_true.reshape(-1, 1)

    ad = []

    for i in range(target.n_var):
        ad.append(anderson_ksamp([X_mcmc[i, :], X_true[i, :]]).statistic)

    if verbose:
        labels = target.get_var_labels()

        print("Anderson-Darling Distance (AD)")

        for i in range(target.n_var):
            print("\t", labels[i], ": ", ad[i])

    return ad


def ess(X_mcmc, target, verbose=False):
    ess = []

    for i in range(target.n_var):
        ess.append(az.ess(X_mcmc[i, :]))

    if verbose:
        labels = target.get_var_labels()

        print("Effective Sample Size (ESS)")

        for i in range(len(ess)):
            print("\t", labels[i], ": ", ess[i])

    return ess


def quantiles(
    X_mcmc,
    target,
    X_true=None,
    sample_size=10000,
    quantiles=[0.025, 0.25, 0.5, 0.75, 0.975],
):
    labels = target.get_var_labels()

    if X_true is None:
        X_true = target.direct_sample(sample_size)

    if target.n_var == 1:
        X_true = X_true.reshape(-1, 1)

    df = pd.DataFrame(columns=[str(q) for q in quantiles])

    for i in range(target.n_var):
        df.loc["True" + labels[i]] = np.quantile(X_true[i, :], quantiles)
        df.loc["MCMC" + labels[i]] = np.quantile(X_mcmc[i, :], quantiles)

    return df
