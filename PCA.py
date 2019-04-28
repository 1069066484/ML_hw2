# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the project of Machine Learning: two PCA approaches.
"""
from sklearn.preprocessing import Normalizer
from numpy import linalg as LA
import numpy as np
import os


class PCA_ling:
    def __init__(self, alg='eig_solve', n_components=None):
        """
        @Param alg: should be one between 'eig_solve' and 'svd', using two approaches to solve PCA.
        @Param n_components: number of components to reduce the data to. If n_components is greater than dimension of
            the data or is not set, then n_components would be adjusted to the right dimensions of the data.
        """
        self.alg = alg
        self.normalizer = None
        self.n_components = n_components
        self.selected_var = None

    def _decentralize(self, data):
        means = np.repeat(np.mean(data,0),data.shape[0],0).reshape(data.shape[1],-1).T
        return data - means

    def _get_XtX(self, data):
        X = self._decentralize(data)
        return [np.matmul( X.T, X) * (1 / (data.shape[0]-1)) ,X]

    def _decide_components(self, data):
        self.n_components = data.shape[1] if self.n_components is None or self.n_components > data.shape[1] else self.n_components

    def _fit_transform_eig_solve(self, data):
        [XtX, X] = self._get_XtX(data)
        w, v = LA.eig(XtX)
        wv = list(zip(w, v.T))
        sorted_wv = sorted(wv, key=lambda x: -abs(x[0]))
        sorted_w = [abs(wv_i[0]) for wv_i in sorted_wv]
        sorted_v = [wv_i[1] for wv_i in sorted_wv]
        selected_vs = np.vstack(sorted_v[:self.n_components]).T
        self.selected_var = np.sum(sorted_w[:self.n_components]) / np.sum(sorted_w)
        return np.matmul(X, selected_vs)

    def _fit_transform_svd(self, data):
        [XtX, X] = self._get_XtX(data)
        U, ds, Vt = LA.svd(X)
        # XtX * V /{by_col} ds == U
        # eig vals of XtX are squared sigs of X
        # U are eigs of XXt, V are eigs of XtX
        selected_vs = Vt[:self.n_components].T
        ds **= 2
        self.selected_var = np.sum(ds[:self.n_components]) / np.sum(ds)
        return np.matmul(X, selected_vs)

    def fit_transform(self, data):
        self._decide_components(data)
        if self.alg == 'eig_solve':
            return self._fit_transform_eig_solve(data)
        elif self.alg == 'svd':
            return self._fit_transform_svd(data)
        else:
            raise Exception("Invalid algorithm alg="+self.alg+"\nalg should be one among eig_solve and svd\n")

    def selected_components(self):
        return self.selected_var


def _read_test_data():
    import data_helper
    import global_defs
    path_test_data = data_helper.npfilename(os.path.join(global_defs.PATH_SAVING_FOLDER, 'pca_test_data'))
    if os.path.exists(path_test_data):
        return np.load(path_test_data)
    test_data = data_helper.read_features(use_dl=True)[:20,:50]
    np.save(path_test_data, test_data)
    return test_data


def _test_PCA_ling():
    data = _read_test_data()
    n_components = 3
    print("ori_data.shape=", data.shape,"  n_components=", n_components)

    for alg in ['eig_solve', 'svd']:
        pca_ling = PCA_ling(alg=alg, n_components=n_components)
        reduced1 = pca_ling.fit_transform(data)
        print(alg, reduced1.shape, pca_ling.selected_components())
    #print(reduced1)

    from sklearn.decomposition import PCA
    pca_sklearn = PCA(n_components=n_components)
    reduced1 = pca_sklearn.fit_transform(data)
    print(reduced1.shape, np.sum(pca_sklearn.explained_variance_ratio_[:n_components]))
    #print(reduced1)


if __name__ == '__main__':
    _test_PCA_ling()

