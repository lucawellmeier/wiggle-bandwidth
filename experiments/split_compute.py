import time
import numpy as np
from sklearn.metrics import mean_squared_error
from shared import MnistBinaryDataset, laplace_kernel, WiggledScaleKernelLS, WiggleSearchScaleKernelLS, get_search_gammas, get_result_and_figure_path

if __name__ == '__main__':
    Ntr = 1000
    Nte = 1000
    Xtr,Ytr, Xte,Yte = MnistBinaryDataset(48888).sample(4, 9, [Ntr, Ntr])
    splits = np.linspace(0.3, 0.95, num=20)
    gamma0 = 1/784
    search_gammas = get_search_gammas(gamma0, 2, 15)
    best_gammas = []
    test_mses = []
    sample_mses = []
    wiggle_mses = []
    fit_times = []
    wiggle_times = []

    baseline = WiggledScaleKernelLS(laplace_kernel, gamma0).fit(Xtr,Ytr)
    baseline_mse = mean_squared_error(baseline.predict(Xte), Yte)
    baseline_sample_mse = mean_squared_error(baseline.predict(Xtr), Ytr)
    print('fit baseline model in {:.2f} seconds'.format(baseline.fit_time))

    for split in splits:
        model = WiggleSearchScaleKernelLS(laplace_kernel, gamma0, search_gammas, split)
        model.fit(Xtr, Ytr)
        fit_times.append(model.fit_time)
        wiggle_times.append(model.wiggle_time)
        best_gammas.append(model.best_gamma)
        wiggle_mses.append(model.best_mse)
        test_mses.append(mean_squared_error(Yte, model.predict(Xte)))
        sample_mses.append(mean_squared_error(Ytr, model.predict(Xtr)))
        print('finished train split at {:.1f}% with total training time of {:.2f} seconds'.format(split*100.0, fit_times[-1] + wiggle_times[-1]))

    resFile,_ = get_result_and_figure_path()
    np.savez(resFile, 
        baseline=baseline_mse, baseline_sample=baseline_sample_mse,
        splits=splits, best_gammas=best_gammas,
        test_mses=test_mses, sample_mses=sample_mses, wiggle_mses=wiggle_mses,
        fit_times=fit_times, wiggle_times=wiggle_times, baseline_fit_time=baseline.fit_time)