from shared import get_result_and_figure_path
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    resFile, figFile = get_result_and_figure_path()
    results = np.load(resFile)
    base_gammas = results['base_gammas']
    min_std_mse = np.argmin(results['standard_mses'])

    plt.figure(figsize=(12, 6))
    for j in range(base_gammas.size):
        min_wig_mse = np.argmin(results['wiggled_mses'][j,:])
        plt.subplot(2, 3, j+1)
        plt.title('$\gamma_0 = {}$'.format(base_gammas[j]))
        plt.xlabel('$\gamma$')
        plt.ylabel('MSE')
        plt.plot(results['wiggle_gammas'], results['standard_mses'], label='standard')
        plt.scatter([results['wiggle_gammas'][min_std_mse]], [results['standard_mses'][min_std_mse]], c='blue')
        plt.plot(results['wiggle_gammas'], results['wiggled_mses'][j,:], label='wiggled')
        plt.scatter([results['wiggle_gammas'][min_wig_mse]], [results['wiggled_mses'][j,min_wig_mse]], c='orange')
        plt.axvline(base_gammas[j], c='green')
        plt.legend()
    plt.tight_layout()
    plt.savefig(figFile)