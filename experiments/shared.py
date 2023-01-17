import os
import time
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
import numpy as np



MNIST_PATH = 'build/mnist_784.npz'

class MnistBinaryDataset:

    def __init__(self, seed):
        if not os.path.exists(MNIST_PATH):
            mnist = fetch_openml('mnist_784', parser='auto')
            self.allX = mnist.data.to_numpy()
            self.allY = mnist.target.astype(int).to_numpy()
            np.savez(MNIST_PATH, X=self.allX, Y=self.allY)
        else:
            archive = np.load(MNIST_PATH)
            self.allX = archive['X']
            self.allY = archive['Y']
        self.dim = self.allX.shape[1]
        self.rng = np.random.default_rng(seed)

    def sample(self, digitA, digitB, split):
        assert(np.all(np.array(split) > 0))
        relevant = np.logical_or(self.allY == digitA, self.allY == digitB)
        N = np.sum(split)
        assert(np.sum(relevant) >= N)

        perm = self.rng.permutation(N)
        X = self.allX[relevant][:N][perm]
        Y = self.allY[relevant][:N][perm].reshape(-1,1)
        Y[Y == digitA] = -1
        Y[Y == digitB] = 1

        curPos = 0
        r = []
        for p in split:
            r.append(X[curPos:curPos+p])
            r.append(Y[curPos:curPos+p])
            curPos += p
        return r



def laplace_kernel(gamma,X0,X1=None):
    if X1 is None: X1 = X0
    D = distance_matrix(X0, X1, p=2)
    return np.exp(-gamma * D)

def gaussian_kernel(gamma,X0,X1=None):
    if X1 is None: X1 = X0
    D = distance_matrix(X0, X1, p=2)
    return np.exp(-gamma * D**2)



class WiggledScaleKernelLS:

    def __init__(self, kernel_func, base_gamma, wiggled_gamma=None):
        self.kernel_func = kernel_func
        self.base_gamma = base_gamma
        self.wiggled_gamma = wiggled_gamma if wiggled_gamma is not None else base_gamma
        self.model = KernelRidge(alpha=0, gamma=base_gamma, kernel='precomputed')

    def fit(self, Xtr, Ytr):
        Ktr = self.kernel_func(self.base_gamma, Xtr)
        self.model.fit(Ktr, Ytr)
        self.Xtr = Xtr
        return self

    def predict(self, Xte):
        Kte = self.kernel_func(self.wiggled_gamma, Xte, self.Xtr)
        return self.model.predict(Kte)



class WiggleSearchScaleKernelLS:

    def __init__(self, kernel_func, gamma0, search_gammas, split):
        self.kernel_func = kernel_func
        self.gamma0 = gamma0
        self.search_gammas = search_gammas
        self.split = split
        self.model = WiggledScaleKernelLS(kernel_func=kernel_func, base_gamma=gamma0)
        self.best_gamma = None

    def fit(self, Xtr, Ytr):
        Ntr = Xtr.shape[0]
        Ntr0 = round(self.split * Ntr)
        Xtr0 = Xtr[:Ntr0]
        Ytr0 = Ytr[:Ntr0]
        Xtr1 = Xtr[Ntr0:]
        Ytr1 = Ytr[Ntr0:]
        self.model.fit(Xtr0, Ytr0)

        mses = []
        for gamma in self.search_gammas:
            self.model.wiggled_gamma = gamma
            Ypr1 = self.model.predict(Xtr1)
            mses.append(mean_squared_error(Ytr1, Ypr1))
        i = np.argmin(np.array(mses))
        self.best_gamma = self.search_gammas[i]
        self.model.wiggled_gamma = self.best_gamma
        self.best_mse = mses[i]
        return self

    def predict(self, Xte):
        return self.model.predict(Xte)



def get_search_gammas(base_gamma, lo, n):
    baseexp = np.log10(base_gamma)
    logs = np.logspace(start=baseexp, stop=baseexp-lo, num=n+1)
    return np.sort(np.concatenate([base_gamma - logs[1:], base_gamma + logs]))

class IteratedWiggleSearchScaleKernelLS:

    def __init__(self, kernel_func, gamma0, n_iter=10, train_split=0.85, search_n=25, search_loexp=1):
        self.kernel_func = kernel_func
        self.gamma0 = gamma0
        self.n_iter = n_iter
        self.train_split = train_split
        self.search_n = search_n
        self.search_loexp = search_loexp

    def fit(self, Xtr, Ytr):
        self.gamma_hist = []
        self.model_hist = []
        self.fit_times = []
        cur_gamma = self.gamma0
        for it in range(0,self.n_iter):
            startTime = time.time()
            search_gammas = get_search_gammas(cur_gamma, self.search_loexp, self.search_n)
            self.model_hist.append(WiggleSearchScaleKernelLS(self.kernel_func, cur_gamma, search_gammas, self.train_split))
            self.model_hist[-1].fit(Xtr, Ytr)
            cur_gamma = self.model_hist[-1].best_gamma
            self.gamma_hist.append(cur_gamma)
            self.fit_times.append(time.time() - startTime)
            print('iteration {}/{} done in {} seconds'.format(it + 1, self.n_iter, self.fit_times[-1]))

    def predict(self, Xte):
        self.model_hist[-1].predict(Xte)