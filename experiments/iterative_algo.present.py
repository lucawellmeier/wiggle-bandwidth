from shared import get_result_and_figure_path
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    resFile, figFile = get_result_and_figure_path()
    results = np.load(resFile)
    n_iter = len(results['refit'])
    iters = list(range(n_iter))

    fig,axs = plt.subplots(1, 2)
    fig.set_size_inches(9,4)
    fig.suptitle('Average fit time: {:.2f} seconds'.format(np.mean(np.array(results['fit_times']))))
    axs[0].set_title('MSE history')
    axs[0].plot(iters, results['refit'], label='refit with best bandwidth')
    axs[0].plot(iters, results['wiggle'], label='with best wiggled bandwidth')
    axs[0].set_xlabel('iteration')
    axs[0].set_ylabel('MSE')
    axs[0].legend()
    axs[1].set_title('bandwidth history')
    axs[1].plot(iters, results['gamma_hist'])
    axs[1].set_xlabel('iteration')
    axs[1].set_ylabel('$\gamma$')

    fig.tight_layout()
    fig.savefig(figFile)