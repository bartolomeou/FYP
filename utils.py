import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import acf

import wesanderson as wes

sns.set_palette(sns.color_palette(wes.film_palette('Darjeeling Limited')))


def traceplot(samples, w=6, h=4, n_col=2, overlay=True):
    # samples: (#components, #iterations)
    
    n_component = samples.shape[0]
    n_iter = samples.shape[1]

    if overlay:
        fig, ax = plt.subplots(figsize=(6,4))
        
        for i in range(n_component):
            sns.lineplot(x=range(1, n_iter+1), y=samples[i, :], label=rf'$X_{i+1}$', linewidth=0.5, legend=False)

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
        n_row = np.ceil(n_component / n_col).astype(int)

        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(w*n_col, h*n_row))

        for i in range(n_component):
            ax = axes.flat[i]
            
            sns.lineplot(x=range(1, n_iter+1), y=samples[i, :], ax=ax, linewidth=0.5)
            
            # Set labels
            ax.set_xlabel('t')
            ax.set_ylabel(fr'$X_{i+1}(t)$')

        # Delete any unused axes (if there are fewer features than subplot slots)
        for i in range(n_component, len(axes.flat)):
            fig.delaxes(axes.flat[i])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def acfplot(samples, lags=50, w=6, h=4, n_col=2, overlay=True):
    # samples: (#components, #iterations)
    n_component = samples.shape[0]

    if overlay:
        fig, ax = plt.subplots(figsize=(6,4))
        
        for i in range(n_component):
            acf_values = acf(samples[i, :], nlags=lags, adjusted=True)
            sns.scatterplot(x=range(0, lags+1), y=acf_values, label=rf'$X_{i+1}$', size=0.5, legend=False)

        # Set labels
        ax.set_xlabel('k')
        ax.set_ylabel(r'$\text{Corr}(X_i(t), X_i(t+k))$')

        # Create figure-level legend outside the plot
        fig.legend(loc='upper left', bbox_to_anchor=(0.9, 0.9), frameon=False)

        # Modify the line width in the legend
        legend = fig.legends[0]
        for line in legend.get_lines():
            line.set_linewidth(1)
        
        plt.show()
    else:
        n_row = np.ceil(n_component / n_col).astype(int)

        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(w*n_col, h*n_row))

        for i in range(n_component):
            ax = axes.flat[i]
            
            acf_values = acf(samples[i, :], nlags=lags, adjusted=True)

            sns.scatterplot(x=range(0, lags+1), y=acf_values, ax=ax, size=0.5, legend=False)
            
            # Set labels
            ax.set_xlabel('k')
            ax.set_ylabel(f'$\\text{{Corr}}(X_{{{i+1}}}(t), X_{{{i+1}}}(t+k))$')

        # Delete any unused axes (if there are fewer features than subplot slots)
        for i in range(n_component, len(axes.flat)):
            fig.delaxes(axes.flat[i])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()