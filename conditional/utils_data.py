import numpy as np


# A class to sample from simulated data
class TrueSampler:
    def __init__(self, sigma=0.2, null_feature=False):
        self.sigma = sigma
        self.null_feature = null_feature

    def sample(self, n=1000, return_mean=False):
        num_features = 7 if not self.null_feature else 8

        X = np.random.rand(n, num_features)
        e = np.random.randn(n)

        mean_vector = (
            8
            + X[:, 0] ** 2
            + X[:, 1] * X[:, 2]
            + np.cos(X[:, 3])
            + np.exp(X[:, 4] * X[:, 5])
            + 0.1 * X[:, 6]
        )

        # test a challenging model #
        # self.sigma will not be used in this case
        sigma_x = 0.4 * X[:, 0]
        y = mean_vector + sigma_x * e
        # end of test #

        print("X shape: ", X.shape)
        print("y shape: ", y.shape)

        if return_mean:
            return X, y, mean_vector, sigma_x
        else:
            return X, y
