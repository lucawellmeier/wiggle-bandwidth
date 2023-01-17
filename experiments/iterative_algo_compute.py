import numpy as np
from sklearn.metrics import mean_squared_error
from shared import MnistBinaryDataset, laplace_kernel, IteratedWiggleSearchScaleKernelLS, get_result_and_figure_path

if __name__ == '__main__':
    Ntr = 2000
    Nte = 2000
    Xtr,Ytr, Xte,Yte = MnistBinaryDataset(3281).sample(4, 9, [Ntr, Ntr])
    gamma0 = 1/784
    n_iter = 10
    search_n = 20
    search_loexp = 2
    split = 0.85

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
    
    resFile,_ = get_result_and_figure_path()
    np.savez(resFile, refit=refit_mses, wiggle=wiggle_mses, gamma_hist=model.gamma_hist)