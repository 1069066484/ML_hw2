# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the project of Machine Learning: dimension reduction with FA - EM algorithm.
"""
import FA_dataset
from sklearn.decomposition import FactorAnalysis
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


class GaussianPdf:
    def __init__(self, mean, sigma2):
        det_sigma2_sqrt2 = np.sqrt(np.linalg.det(sigma2))
        #print("sigma2=",sigma2)
        #print("det_sigma2_sqrt2=",det_sigma2_sqrt2)
        self.mean = mean
        fact1 = (2 * np.pi) ** (len(mean) / 2)
        self.fact = 1.0/(det_sigma2_sqrt2 * fact1)
        self.sigma2_I = np.mat(sigma2).I

    def pdf(self, x):
        return self._pdf1(x) if len(x.shape) == 1 else self._pdf2(x)

    def _pdf1(self, x):
        x_minus_mu = x - self.mean
        x_minus_mu = x_minus_mu.reshape([len(x_minus_mu), -1])
        #x_minus_mu = x_minus_mu.t
        return np.array(self.fact * np.exp(-0.5 * np.matmul( x_minus_mu.T, np.matmul(self.sigma2_I , x_minus_mu))))[0][0]

    def _pdf2(self, x):
        x_minus_mu = x - np.vstack([ self.mean for _ in range(x.shape[0])])
        x_minus_mu = x_minus_mu.reshape([len(x_minus_mu), -1]).T
        #print("x_minus_mu.shape=", x_minus_mu.shape)#(3, 2)
        #print("self.sigma2_I.shape=", self.sigma2_I.shape)# (3, 3)
        return np.diag(self.fact * np.exp(-0.5 * np.matmul( x_minus_mu.T, np.matmul(self.sigma2_I , x_minus_mu))))

    def sum_log_pdf(self, x):
        pdfs = self._pdf2(x)
        return np.sum(np.log(pdfs))


def pdf(x, mean, sigma2):
    x_minus_mu = x - mean
    x_minus_mu = x_minus_mu.reshape([len(x_minus_mu), -1])
    det_sigma2 = np.linalg.det(sigma2)
    det_sigma2_sqrt2 = np.sqrt(det_sigma2)
    fact1 = (2 * np.pi) ** (len(mean) / 2)
    #print("np.mat(sigma2).I.shape=", np.mat(sigma2).I.shape)
    #print("x_minus_mu.shape=", x_minus_mu.shape)
    return 1.0/(det_sigma2_sqrt2 * fact1) * np.exp(-0.5 * np.matmul( x_minus_mu.T, np.matmul( np.mat(sigma2).I, x_minus_mu)))


def plt_show_result(ncomps_aic_bic):
    print(ncomps_aic_bic)
    plt.plot(ncomps_aic_bic[0], ncomps_aic_bic[1], linestyle="--")
    plt.plot(ncomps_aic_bic[0], ncomps_aic_bic[2])
    plt.plot(ncomps_aic_bic[0], ncomps_aic_bic[3])
    plt.xlabel('Selected m')
    plt.ylabel('criteria')
    plt.legend(['LLL','BIC', 'AIC'], loc = 'upper left')
    plt.title('selected m - AIC,BIC Relation')
    plt.show()


def selectedm_criteria_test():
    N=100
    n=10
    m=3
    sigma2=0.1
    mu=0
    [X, Y, A] = FA_dataset.read_dataset(N,n,m,sigma2, mu)
    ncomps_aic_bic = [[],[],[],[]]
    for n_components in range(1,8):
        fa = FactorAnalysis(n_components=n_components, tol=0.0001, max_iter=1000)
        fa.fit(X)
        gc = GaussianPdf(fa.mean_, fa.get_covariance())
        # fa.loglike_[-1] == gc.sum_log_pdf(X)
        loglikelihood = gc.sum_log_pdf(X)
        d = n * n_components + X.shape[1] * 2
        ncomps_aic_bic[0].append(n_components)
        ncomps_aic_bic[1].append(loglikelihood)
        ncomps_aic_bic[2].append(loglikelihood - d)
        ncomps_aic_bic[3].append(loglikelihood - 0.5 * d * np.log(N))
        print(n_components, loglikelihood - d, loglikelihood - 0.5 * d * np.log(N))
    plt_show_result(ncomps_aic_bic)


if __name__ == '__main__':
    selectedm_criteria_test()

"""
FactorAnalysis
X = AY+Mu+e
fa.components_.T: A
fa.fit_transform(X): Y
fa.noise_variance_: noise_sigma2
fa.get_covariance(): noise_variance_ + AA.T
"""