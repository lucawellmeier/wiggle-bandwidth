import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.special import zeta
from shared import laplace_kernel, WiggledScaleKernelLS, get_result_and_figure_path

def f(x): return zeta(x) * np.sin(20*x)

if __name__ == '__main__':
    rng = np.random.default_rng(2321)
    Ntr = 30
    Nte = 1000

    Xtr = rng.uniform(1.5, 3, Ntr).reshape(-1,1)
    Ytr = f(Xtr)
    Xte = np.linspace(1.5, 3, Nte).reshape(-1,1)
    Yte = f(Xte)

    base_gammas = np.array([11, 13, 17, 20, 25, 30])
    wiggle_gammas = np.linspace(10, 50, num=100)
    wiggled_mses = np.zeros((base_gammas.size, wiggle_gammas.size))

    standard_mses = np.zeros(wiggle_gammas.size)
    for i in range(wiggle_gammas.size):
        model = WiggledScaleKernelLS(kernel_func=laplace_kernel, base_gamma=wiggle_gammas[i])
        model.fit(Xtr, Ytr)
        standard_mses[i] = mean_squared_error(Yte, model.predict(Xte))
    min_std_mse = np.argmin(standard_mses)

    for j in range(base_gammas.size):
        model = WiggledScaleKernelLS(kernel_func=laplace_kernel, base_gamma=base_gammas[j]).fit(Xtr, Ytr)
        for i in range(wiggle_gammas.size):
            model.wiggled_gamma = wiggle_gammas[i]
            Ypr = model.predict(Xte)
            wiggled_mses[j,i] = mean_squared_error(Yte, Ypr)
    
    resFile,_ = get_result_and_figure_path()
    np.savez(resFile, 
        base_gammas=base_gammas, wiggle_gammas=wiggle_gammas, 
        standard_mses=standard_mses, min_std_mse=min_std_mse,
        wiggled_mses=wiggled_mses)