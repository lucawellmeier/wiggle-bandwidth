import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from code.shared import MnistBinaryDataset, laplace_kernel, IteratedWiggleSearchScaleKernelLS

RES_FILE = 'build/results/iterative_algo.npz'
FIG_FILE = 'build/figures/iterative_algo.png'

Ntr = 200
Nte = 500
Xtr,Ytr, Xte,Yte = MnistBinaryDataset(3281).sample(4, 9, [Ntr, Ntr])

gamma0 = 1/784
n_iter = 8
search_n = 5
search_loexp = 2
split = 0.85

def compute():
    model = IteratedWiggleSearchScaleKernelLS(kernel_func=laplace_kernel, 
        gamma0=gamma0, n_iter=n_iter, search_n=search_n, search_loexp=search_loexp, train_split=split)
    model.fit(Xtr, Ytr)

    iters = list(range(n_iter))
    refit_mses = []
    wiggle_mses = []
    gamma_hist = []
    for it in iters:
        Ypr = model.model_hist[it].predict(Xte)
        refit_mses.append(mean_squared_error(Yte, Ypr))
        wiggle_mses.append(model.model_hist[it].best_mse)
    
    np.savez(RES_FILE, refit=refit_mses, wiggle=wiggle_mses, gamma_hist=model.gamma_hist)

def figure(show=False):
    iters = list(range(n_iter))
    results = np.load(RES_FILE)

    fig,axs = plt.subplots(1, 2)
    fig.set_size_inches(9,3)
    axs[0].set_title('MSE history')
    axs[0].plot(iters, results['refit'], label='refit with best bandwidth')
    axs[0].plot(iters, results['wiggle'], label='with best wiggled bandwidth')
    axs[0].legend()
    axs[1].set_title('$\gamma$ history')
    axs[1].plot(iters, results['gamma_hist'])

    if show:
        plt.show()
    else:
        fig.savefig(FIG_FILE)