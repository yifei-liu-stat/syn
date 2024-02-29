import numpy as np


# A class to sample from simulated data
class TrueSampler:
    def __init__(self, sigma=0.2, null_feature=False):
        self.sigma = sigma
        self.null_feature = null_feature

    def sample(self, n=1000):
        num_features = 7 if not self.null_feature else 8

        X = np.random.rand(n, num_features)
        e = np.random.randn(n)

        y = (
            8
            + X[:, 0] ** 2
            + X[:, 1] * X[:, 2]
            + np.cos(X[:, 3])
            + np.exp(X[:, 4] * X[:, 5])
            + 0.1 * X[:, 6]
            + self.sigma * e
        )

        print("X shape: ", X.shape)
        print("y shape: ", y.shape)
        return X, y
