import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import acf

import wesanderson as wes


sns.set_palette(sns.color_palette(wes.film_palette('The Life Aquatic with Steve Zissou')))


def traceplot(X, w=6, h=4, n_col=2, overlay=True):
    # X: (#components, #iterations)
    n_dim = X.shape[0]
    n_iter = X.shape[1]

    if overlay:
        sns.set_palette(sns.color_palette(wes.film_palette('Darjeeling Limited')))

        fig, ax = plt.subplots(figsize=(6,4))
        
        for i in range(n_dim):
            sns.lineplot(x=range(1, n_iter+1), y=X[i, :], label=rf'$X_{i+1}$', linewidth=0.5, legend=False)

        # Set labels
        ax.set_xlabel('t')
        ax.set_ylabel(r'$X_i(t)$')

        # Create figure-level legend outside the plot
        fig.legend(loc='upper left', bbox_to_anchor=(0.9, 0.9), frameon=False)

        # Modify the line width in the legend
        legend = fig.legends[0]
        for line in legend.get_lines():
            line.set_linewidth(1)
        
        plt.show()
    else:
        sns.set_palette(sns.color_palette(wes.film_palette('The Life Aquatic with Steve Zissou')))

        n_row = np.ceil(n_dim / n_col).astype(int)

        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(w*n_col, h*n_row))

        for i in range(n_dim):
            ax = axes.flat[i]
            
            sns.lineplot(x=range(1, n_iter+1), y=X[i, :], ax=ax, linewidth=0.5)
            
            # Set labels
            ax.set_xlabel('t')
            ax.set_ylabel(fr'$X_{i+1}(t)$')

        # Delete any unused axes (if there are fewer features than subplot slots)
        for i in range(n_dim, len(axes.flat)):
            fig.delaxes(axes.flat[i])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def acfplot(X, lags=50, w=6, h=4, n_col=2, overlay=True):
    # X: (#components, #iterations)
    n_dim = X.shape[0]

    lags = min(lags, X.shape[1])

    if overlay:
        sns.set_palette(sns.color_palette(wes.film_palette('Darjeeling Limited')))

        fig, ax = plt.subplots(figsize=(6,4))
        
        for i in range(n_dim):
            acf_values = acf(X[i, :], nlags=lags, adjusted=True)
            sns.scatterplot(x=range(0, lags+1), y=acf_values, label=rf'$X_{i+1}$', size=0.5, legend=False)

        # Set labels
        ax.set_xlabel('k')
        ax.set_ylabel(r'$\mathrm{Corr}(X_{i}(t),\,X_{i}(t+k))$')

        # Create figure-level legend outside the plot
        fig.legend(loc='upper left', bbox_to_anchor=(0.9, 0.9), frameon=False)

        # Modify the line width in the legend
        legend = fig.legends[0]
        for line in legend.get_lines():
            line.set_linewidth(1)
        
        plt.show()
    else:
        sns.set_palette(sns.color_palette(wes.film_palette('The Life Aquatic with Steve Zissou')))

        n_row = np.ceil(n_dim / n_col).astype(int)

        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(w*n_col, h*n_row))

        for i in range(n_dim):
            ax = axes.flat[i]
            
            acf_values = acf(X[i, :], nlags=lags, adjusted=True)

            sns.scatterplot(x=range(0, lags+1), y=acf_values, ax=ax, size=0.5, legend=False)
            
            # Set labels
            ax.set_xlabel('k')
            ax.set_ylabel(f'$\\text{{Corr}}(X_{{{i+1}}}(t), X_{{{i+1}}}(t+k))$')

        # Delete any unused axes (if there are fewer features than subplot slots)
        for i in range(n_dim, len(axes.flat)):
            fig.delaxes(axes.flat[i])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def pairplot(X, n1, n2, sampler_names=None):
    column_labels = ['X_{1}']

    for j in range(1, n2+1):
        for i in range(2, n1+1):
            column_labels.append(f'X_{{{j},{i}}}')

    if isinstance(X, list):
        sns.set_palette(sns.color_palette(wes.film_palette('Darjeeling Limited')))
        
        combined_df = pd.DataFrame()

        if sampler_names is None:
            sampler_names = [str(i) for i in range(1, len(X)+1)]

        for s in range(len(X)):
            df = pd.DataFrame(data=X[s].T, columns=column_labels)
            df['Sampler'] = sampler_names[s]
            combined_df = pd.concat([combined_df, df], ignore_index=True)

        grid = sns.pairplot(combined_df, hue='Sampler', diag_kind='kde', plot_kws={'s': 5, 'alpha': 0.2})
        
        # Make legend markers fully opaque
        for line in grid.legend.get_lines(): 
            line.set(alpha=1.0)
            line.set(ms=5)

        plt.show()
    else:
        sns.set_palette(sns.color_palette(wes.film_palette('The Life Aquatic with Steve Zissou')))

        df = pd.DataFrame(data=X.T, columns=column_labels)
    
        sns.pairplot(df, diag_kind='kde', plot_kws={'s': 10, 'alpha': 0.2})