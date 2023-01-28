from shared import get_result_and_figure_path
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    resFile, figFile = get_result_and_figure_path()
    results = np.load(resFile)
    alphas = results['alphas']

    plt.plot([alphas[0], alphas[-1]], 2*[results['baseline_mse']], label='baseline')
    plt.plot(alphas, results['tikhonov_mses'], label='tikhonov')
    plt.plot([alphas[0], alphas[-1]], 2*[results['wiggled_mse']], label='wiggled')
    plt.plot(alphas, results['wiggled_tikhonov_mses'], label='wiggled tikhonov')
    plt.xscale('log')
    plt.xlabel('$\\alpha$')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig(figFile)