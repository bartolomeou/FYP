def ess(X, n1, n2):
    idx = ['X_{1}']

    for j in range(1, n2+1):
        for i in range(2, n1+1):
            idx.append(f'X_{{{j},{i}}}')

    print('ESS for each component:')

    for i in range(X.shape[0]):
        print('\t', idx[i], ': ', az.ess(X[i, :]))


def compare_quantiles(X_mcmc, n1, n2, n_iter=10000):
    quantiles = (0.025, 0.25, 0.5, 0.75, 0.975)

    idx = ['X_{1}']

    for j in range(1, n2+1):
        for i in range(2, n1+1):
            idx.append(f'X_{{{j},{i}}}')

    X_true = sample_rosenbrock(n_iter, n1, n2)

    df = pd.DataFrame(columns=['0.025', '0.25', '0.5', '0.75', '0.975'])

    for i in range(X_true.shape[0]):
        df.loc['True ' + idx[i]] = np.quantile(X_true[i, :], quantiles)
        df.loc['MCMC ' + idx[i]] = np.quantile(X_mcmc[i, :], quantiles)

    return df