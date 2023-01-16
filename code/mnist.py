import os
import numpy as np
from sklearn.datasets import fetch_openml

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

if __name__ == '__main__':
    MnistBinaryDataset(0)