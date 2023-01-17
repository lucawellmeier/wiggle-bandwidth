from shared import get_result_and_figure_path
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    resFile, figFile = get_result_and_figure_path()
    results = np.load(resFile)
    splits = results['splits']

    fig,axs = plt.subplots(2, 2)
    fig.set_size_inches(10,10)

    axs[0,0].set_title('sample MSE')
    axs[0,0].plot([splits[0],splits[-1]], 2*[results['baseline_sample']], label='baseline', c='black')
    axs[0,0].plot(splits, results['sample_mses'], label='wiggle search')
    axs[0,0].set_xlabel('train split')
    axs[0,0].set_ylabel('MSE')
    axs[0,0].legend()

    axs[0,1].set_title('test MSE')
    axs[0,1].plot([splits[0],splits[-1]], 2*[results['baseline']], label='baseline', c='black')
    axs[0,1].plot(splits, results['test_mses'], label='wiggle search on test')
    axs[0,1].plot(splits, results['wiggle_mses'], label='wiggle search on internal test')
    axs[0,1].set_xlabel('train split')
    axs[0,1].set_ylabel('MSE')
    axs[0,1].legend()

    axs[1,0].set_title('chosen bandwidths')
    axs[1,0].plot(splits, results['best_gammas'])
    axs[1,0].set_yscale('log')
    axs[1,0].set_xlabel('train split')
    axs[1,0].set_ylabel('$\gamma$')

    axs[1,1].set_title('fit times')
    axs[1,1].plot([splits[0],splits[-1]], 2*[results['baseline_fit_time']], label='baseline', c='black')
    total = np.array(results['wiggle_times']) + np.array(results['fit_times'])
    wiggle = np.array(results['wiggle_times'])
    axs[1,1].fill_between(splits, total, label='least squares')
    axs[1,1].fill_between(splits, wiggle, label='wiggle')
    axs[1,1].plot(splits, total)
    axs[1,1].set_xlabel('train split')
    axs[1,1].set_ylabel('seconds')
    axs[1,1].legend()

    fig.savefig(figFile)