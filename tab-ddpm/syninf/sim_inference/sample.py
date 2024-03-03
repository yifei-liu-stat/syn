import numpy as np
import pandas as pd


# A class to sample from simulated data
class TrueSampler:
    def __init__(self, sigma=0.2, null_feature=False):
        self.sigma = sigma
        self.null_feature = null_feature

    def sample(self, n=1000, return_df=False):
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
        if return_df:
            df = pd.DataFrame(
                np.concatenate([y[:, None], X], axis=1),
                columns=["y"] + [f"num_{i}" for i in range(X.shape[1])],
            )
            print("df shape: ", df.shape)
            return df
        else:
            print("X shape: ", X.shape)
            print("y shape: ", y.shape)
            return X, y
