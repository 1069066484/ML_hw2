# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the project of Machine Learning: dimension reduction with FA - data-set generation.
"""
import numpy as np
from collections import Iterable
import os
import global_defs
import data_helper
import pickle


def _gen_gaussian(mean, cov, num):
    if not isinstance(cov[0], Iterable):
        cov = np.diag(cov)
    return np.random.multivariate_normal(mean, cov, num, check_valid='raise')


def _gen_gaussian01(dim, num, cov=None, mean=None):
    cov = [1 for _ in range(dim)] if cov is None else [cov for _ in range(dim)]
    mean = [0 for _ in range(dim)] if mean is None else [mean for _ in range(dim)]
    return _gen_gaussian(mean, cov, num)


def _gen_dataset(N=100, n=10, m=3, sigma2=0.1, mu=0):
    ys = _gen_gaussian01(m, N)
    es = _gen_gaussian01(n, N, sigma2)
    A = np.random.random([n ,m])
    print(A.shape, ys.shape, es.shape) #(10, 3) (25, 3) (25, 10)
    xs = np.matmul(ys, A.T) + mu + es
    return [xs, ys, A]


def read_dataset(N=100, n=10, m=3, sigma2=0.1, mu=0):
    data_file = os.path.join(global_defs.PATH_SAVING_FOLDER, 'FA_dataset' +
                             '_' + str(N) + '_' + str(n) + '_' + str(m) + '_' + str(sigma2) + '_' + str(mu))
    data_file = data_helper.pkfilename(data_file)
    if os.path.exists(data_file):
        return pickle.load(open(data_file, 'rb'))
    xsysA = _gen_dataset(N, n, m, sigma2, mu)
    pickle.dump(xsysA, open(data_file, 'wb'))
    return xsysA


def _test_gen_dataset():
    [xs, A] = read_dataset(25, 10, 3)
    print(xs, '\n', A)


if __name__=='__main__':
    _test_gen_dataset()


