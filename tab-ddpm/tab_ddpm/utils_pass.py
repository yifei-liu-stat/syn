import numpy as np
from scipy.stats import norm, uniform
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import dcor
from pprint import pprint
from itertools import product
import torch


# helper function for halton()
# credit: https://gist.github.com/tupui/cea0a91cc127ea3890ac0f002f887bae
def primes_from_2_to(n):
    """Prime number from 2 to n.
    From `StackOverflow <https://stackoverflow.com/questions/2068372>`_.
    :param int n: sup bound with ``n >= 6``.
    :return: primes in 2 <= p < n.
    :rtype: list
    """
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=bool)
    for i in range(1, int(n**0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3 :: 2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3 :: 2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


# helper function for halton()
# credit: https://gist.github.com/tupui/cea0a91cc127ea3890ac0f002f887bae
def van_der_corput(n_sample, base=2):
    """Van der Corput sequence.
    :param int n_sample: number of element of the sequence.
    :param int base: base of the sequence.
    :return: sequence of Van der Corput.
    :rtype: list (n_samples,)
    """
    sequence = []
    for i in range(n_sample):
        n_th_number, denom = 0.0, 1.0
        while i > 0:
            i, remainder = divmod(i, base)
            denom *= base
            n_th_number += remainder / denom
        sequence.append(n_th_number)

    return sequence


# generate Halton sequence with specified size and dimension
# credit: https://gist.github.com/tupui/cea0a91cc127ea3890ac0f002f887bae
# return halton sequence as np.array() with shape [n_sample, dim]
def halton(n_sample, dim):
    """Halton sequence.
    :param int dim: dimension
    :param int n_sample: number of samples.
    :return: sequence of Halton.
    :rtype: array_like (n_samples, n_features)
    """
    big_number = 10
    while "Not enought primes":
        base = primes_from_2_to(big_number)[:dim]
        if len(base) == dim:
            break
        big_number += 1000

    # Generate a sample using a Van der Corput sequence per dimension.
    sample = [van_der_corput(n_sample + 1, dim) for dim in base]
    sample = np.stack(sample, axis=-1)[1:]

    return sample


# (description) generate \mc H_n^d for both d1 and d2 as in Deb's paper
# slightly different with their implementation, see:
# https://github.com/NabarunD/MultiDistFree/blob/master/FinalIndependence.R
# , computestatisticrdcov() funciton they defined
# (input) n, d1, d2: size of the problem
# (output) if d2 is 0, then we have sequence for d1 (or d2) only;
# (cont.)  otherwise, the first d1 columns are for X, and the last d2 are for Y
# (cont.)  also, result is np.array() with shape [n, d1 + d2]
def myhalton(n, d1=1, d2=0):
    if d2 == 0:
        if d1 == 1:
            seq = (np.array(range(n)) + 1) / n
            return seq.reshape(n, 1)
        else:
            return halton(n, d1)

    # define described sequence for d1
    if d1 == 1:
        HX = np.reshape((np.array(range(n)) + 1) / n, (n, 1))
    else:
        HX = halton(n, d1)

    # define described sequence for d2
    if d2 == 1:
        HY = np.reshape((np.array(range(n)) + 1) / n, (n, 1))
    else:
        HY = halton(n, d1 + d2)[:, d1 : (d1 + d2)]

    return np.hstack((HX, HY))


# empirical rank map
# (input) X: np.array of shape [n, d]
# (output) output is of dictionary form, with:
# perm_map: optimal permutation for the assignment problem
# mapfrom:  domain of the rank map, that is, X itself
# mapto:    image of the rank map, that is, optimal row-permutated H
# and all values are of type np.array
def erankmap(X, ref="halton"):
    if len(X.shape) == 1:
        X = X[:, None]

    n = X.shape[0]
    d = X.shape[1]
    if ref == "halton":
        H = myhalton(n, d)
    if ref == "uniform":
        H = np.random.rand(n, d)

    dismat = cdist(X, H, metric="sqeuclidean")
    row_ind, col_ind = linear_sum_assignment(dismat)

    perm_map = np.hstack((row_ind.reshape(n, 1), col_ind.reshape(n, 1)))
    mapfrom = X
    mapto = H[col_ind, :]

    return {"perm_map": perm_map, "mapfrom": mapfrom, "mapto": mapto}


# solve matching problem with square Euclidean distance
def perfect_match(Y, Z):
    n = Y.shape[0]

    dismat = cdist(Y, Z, metric="sqeuclidean")
    row_ind, col_ind = linear_sum_assignment(dismat)

    perm_map = np.hstack((row_ind.reshape(n, 1), col_ind.reshape(n, 1)))
    mapfrom = Y
    mapto = Z[col_ind, :]
    opt_cost = dismat[row_ind, col_ind].sum()
    return {"perm_map": perm_map, "mapfrom": mapfrom, "mapto": mapto, "cost": opt_cost}


# permute Y such that it has the same multivariate ranks as Z
# returns Y_matched and inverse of the permutation map
def perfect_permute(Y, Z):
    deviceY, dtypeY = None, None
    if isinstance(Y, torch.Tensor):
        deviceY = Y.device
        dtypeY = Y.dtype
        Y = Y.detach().cpu().numpy()
    if isinstance(Z, torch.Tensor):
        Z = Z.detach().cpu().numpy()

    rank_perm_Y = erankmap(Y)["perm_map"]
    rank_perm_Z = erankmap(Z)["perm_map"]

    rank_perm_list = [
        [y_map[0], z_map[0]]
        for y_map, z_map in zip(
            sorted(rank_perm_Y, key=lambda x: x[1]),
            sorted(rank_perm_Z, key=lambda x: x[1]),
        )
    ]

    rank_perm = np.array(sorted(rank_perm_list, key=lambda x: x[1]))
    rank_perm_inverse = np.array(sorted(rank_perm_list, key=lambda x: x[0]))

    Y_matched = Y[rank_perm[:, 0]]

    if deviceY is not None:
        Y_matched = torch.from_numpy(Y_matched).to(deviceY, dtype=dtypeY)
        rank_perm_inverse = torch.from_numpy(rank_perm_inverse).to(
            deviceY, dtype=torch.long
        )
    return Y_matched, rank_perm_inverse[:, 1]


# (to be solved) overflow issue of exponential with small tau
def cdf_conv_gaussian_laplace(z=0, tau=0.5):
    if tau <= 0:
        return None
    result = (
        norm.cdf(z)
        - 0.5 * np.exp(0.5 / tau**2 - z / tau) * norm.cdf(z - 1 / tau)
        + 0.5 * np.exp(0.5 / tau**2 + z / tau) * (1 - norm.cdf(z + 1 / tau))
    )
    return np.clip(result, 0, 1)


# perturb Y of Gaussian i.i.d. entries to \tilda{Y} with the same i.i.d. entries by adding Laplace noises
def perturb_pass_gaussian_laplace(Y=None, tau=0.5):
    if tau <= 0 or Y.numel() == 0:
        return Y
    if Y is None:
        return None
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()
    noise = tau * np.random.laplace(size=Y.shape)
    Y_tilde = norm.ppf(cdf_conv_gaussian_laplace(Y + noise))

    return Y_tilde


def cdf_conv_uniform_laplace(z=0, tau=0.5):
    if tau <= 0:
        return None
    if z <= 0:
        result = -0.5 * np.exp(z / tau) * tau * (1 - np.exp(-1 / tau))
    elif z >= 1:
        result = 1 - 0.5 * np.exp(-z / tau) * tau * (np.exp(1 / tau) - 1)
    else:
        result = z - 0.5 * tau * (np.exp(-(1 - z) / tau) - np.exp(-z / tau))
    return np.clip(result, 0, 1)


cdf_conv_uniform_laplace = np.vectorize(cdf_conv_uniform_laplace)


# perturb Y of Uniform i.i.d. entries to \tilda{Y} with the same i.i.d. entries by adding Laplace noises
def perturb_pass_uniform_laplace(Y=None, tau=0.5):
    if tau <= 0 or Y.numel() == 0:
        return Y
    if Y is None:
        return None

    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()
    noise = tau * np.random.laplace(size=Y.shape)
    Y_tilde = uniform.ppf(cdf_conv_uniform_laplace(Y + noise))
    return Y_tilde
