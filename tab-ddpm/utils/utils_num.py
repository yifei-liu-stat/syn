"""Utils for numerical calculations and metrics."""

import numpy as np
import pandas as pd


def symmetrize(m: np.array):
    """Symmetrize a matrix."""
    return (m + m.T) / 2


def matrix_sqrt(m: np.array, tolerance: float = 1e-8):
    """Calculate sqrt of a real semi-positive definite matrix."""

    if not (m == m.T).all():
        print("Warning: matrix is not symmetric, symmetrizing...")
        m = symmetrize(m)
        m = m + tolerance * np.eye(m.shape[0])

    # computing diagonalization
    evalues, evectors = np.linalg.eig(m)

    # zero-out small or negative eigenvalues due to numerical precision
    evalues[evalues < tolerance] = 0

    sqrt_matrix = evectors * np.sqrt(evalues) @ np.linalg.inv(evectors)

    return sqrt_matrix


def calculate_fid(df1: pd.DataFrame, df2: pd.DataFrame, tolerance: float = 1e-8):
    """
    Calculate FID between two pandas dataframes with numerical columns.
    - https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance
    """

    # ensure the dataframes have the same columns
    if set(df1.columns) != set(df2.columns):
        raise ValueError("Dataframes should have the same columns")

    # convert dataframes to numpy arrays
    data1 = df1.values
    data2 = df2.values

    # calculate mean and covariance for both distributions
    mu1, sigma1 = np.mean(data1, axis=0), np.cov(data1, rowvar=False)
    mu2, sigma2 = np.mean(data2, axis=0), np.cov(data2, rowvar=False)

    # compute the square of the distance between the means
    diff = mu1 - mu2
    mean_diff = np.dot(diff, diff.T)

    # calculate the trace of the product of the covariances
    sqrt_sigma1_sigma2 = matrix_sqrt(sigma1 @ sigma2, tolerance)

    # handle possible imaginary numbers due to numerical imprecision
    if np.iscomplexobj(sqrt_sigma1_sigma2):
        sqrt_sigma1_sigma2 = sqrt_sigma1_sigma2.real

    trace_sqrt_product = np.trace(sqrt_sigma1_sigma2)

    # compute FID using the calculated values
    fid = mean_diff + np.trace(sigma1) + np.trace(sigma2) - 2 * trace_sqrt_product

    return fid


# Wasserstein-1/2 distance between empirical distributions
import ot


def wasserstein_1_distance(dfs, dft):
    """
    Calculate the Wasserstein-1 distance between two empirical distributions. References:
    - https://pythonot.github.io/
    - https://pythonot.github.io/quickstart.html#computing-wasserstein-distance
    - https://pythonot.github.io/all.html#ot.emd2
    """
    xs, xt = dfs.values, dft.values

    M = ot.dist(xs, xt, metric="euclidean")
    ns, nt = M.shape
    a = np.ones((ns,)) / ns  # uniform distribution on source samples
    b = np.ones((nt,)) / nt  # uniform distribution on target samples

    return ot.emd2(a, b, M, numItermax=10000000, numThreads="max")


def wasserstein_2_distance(dfs, dft):
    """
    Calculate the Wasserstein-2 distance between two empirical distributions. References:
    - https://pythonot.github.io/
    - https://pythonot.github.io/quickstart.html#computing-wasserstein-distance
    - https://pythonot.github.io/all.html#ot.emd2
    """
    xs, xt = dfs.values, dft.values

    M = ot.dist(xs, xt)
    ns, nt = M.shape
    a = np.ones((ns,)) / ns  # uniform distribution on source samples
    b = np.ones((nt,)) / nt  # uniform distribution on target samples

    return np.sqrt(ot.emd2(a, b, M, numItermax=10000000, numThreads="max"))
