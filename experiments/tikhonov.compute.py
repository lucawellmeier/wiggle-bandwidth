import time
import numpy as np
from sklearn.metrics import mean_squared_error
from shared import MnistBinaryDataset, laplace_kernel, WiggledScaleKernelLS, WiggleSearchScaleKernelLS, get_search_gammas, get_result_and_figure_path

if __name__ == '__main__':
    Ntr = 1000
    Nte = 1000
    Xtr,Ytr, Xte,Yte = MnistBinaryDataset(98).sample(4, 9, [Ntr, Nte])

    gamma0 = 1/784
    alphas = np.logspace(-5, 1, num=15)

    baseline = WiggledScaleKernelLS(laplace_kernel, base_gamma=gamma0)
    baseline.fit(Xtr, Ytr)
    baseline_mse = mean_squared_error(Yte, baseline.predict(Xte))
    wiggled = WiggleSearchScaleKernelLS(laplace_kernel, gamma0, get_search_gammas(gamma0, 2, 20), 0.85)
    wiggled.fit(Xtr, Ytr)
    wiggled_mse = mean_squared_error(Yte, wiggled.predict(Xte))

    tikhonov_mses = []
    wiggled_tikhonov_mses = []
    for i in range(alphas.size):
        alpha = alphas[i]

        startTime = time.time()
        tikh = WiggledScaleKernelLS(laplace_kernel, base_gamma=gamma0, alpha=alpha)
        tikh.fit(Xtr, Ytr)
        tikhonov_mses.append(mean_squared_error(Yte, tikh.predict(Xte)))

        wtikh = WiggleSearchScaleKernelLS(laplace_kernel, gamma0, get_search_gammas(gamma0, 2, 20), 0.85, alpha=alpha)
        wtikh.fit(Xtr, Ytr)
        wiggled_tikhonov_mses.append(mean_squared_error(Yte, wtikh.predict(Xte)))

        print('finished alpha {}/{} in {:.2f} seconds'.format(i+1, alphas.size, time.time() - startTime))

    resFile,_ = get_result_and_figure_path()
    np.savez(resFile, alphas=alphas, baseline_mse=baseline_mse, wiggled_mse=wiggled_mse, 
        tikhonov_mses=tikhonov_mses, wiggled_tikhonov_mses=wiggled_tikhonov_mses)