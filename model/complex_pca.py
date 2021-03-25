# -*- coding: utf-8 -*-
"""
@Time ： 2021/3/25 20:48
@Auth ： JsHou
@Email : 137073284@qq.com
@File ：complex_pca.py

"""

import cmath
import math

import numpy as np


class ComplexPCA:
    def __init__(self, n_components=None, threshold=None):
        self.n_components = n_components
        self.threshold = threshold

    def preprocess(self, X, y=None):
        return X

    def inverse_data_process(self, X, y=None):
        return X

    def fit(self, X, y=None):
        X = X.copy()
        self._fit(X)
        return self

    def _fit(self, X):
        """Fit the model by computing eigenvalue decomposition on X * X.H"""
        Z = self.preprocess(X)
        n_samples, n_features = Z.shape
        n_components = min(n_samples, n_features) if self.n_components is None else self.n_components

        # Use the eigenvalue decomposition method to obtain the transformation matrix B_H from x to y
        Z_H = np.conj(Z).T
        K = Z @ Z_H
        w, v = np.linalg.eig(K)  # The eigenvalues w are not ordered
        order = np.argsort(w)
        w = np.sort(w)
        v = v[:, order]
        w1 = np.flip(w).real
        v1 = np.fliplr(v)

        U_m = v1[:, :n_components]
        Lambda_sqrt_minus_2 = np.diag(np.float_power(w1[:n_components], -0.5))
        B = Z_H @ U_m @ Lambda_sqrt_minus_2
        B_H = np.conj(B).T

        # Get variance explained by eigenvalues
        w1 = np.sqrt(w1)
        try:
            total_variance = w1.sum()
            explained_variance_ratio_ = w1 / total_variance
        except ZeroDivisionError:
            print('all eigenvalue of covariance of X are 0.')
            return ZeroDivisionError

        # Calculate the cumulative contribution rate of variance
        if self.n_components == min(n_samples, n_features) and self.threshold is not None:
            explained_variance_ratio_sum_ = 0
            for t, e in enumerate(explained_variance_ratio_):
                explained_variance_ratio_sum_ += e
                if explained_variance_ratio_sum_ >= self.threshold:
                    n_components = t + 1
                    break
        else:
            explained_variance_ratio_sum_ = sum(explained_variance_ratio_[:n_components])

        # Add instance attributes
        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = B_H[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        self.explained_variance_ratio_sum = explained_variance_ratio_sum_
        self.B_H = B_H[:n_components, :]
        self.B = B[:, :n_components]
        self.w = w

        return B_H, B

    def transform(self, Z):
        Z = Z.copy()
        Z = self.preprocess(Z)
        return (self.B_H @ Z.T).T

    def inverse_transform(self, Y):
        Y = Y.copy()
        X = (self.B @ Y.T).T
        return self.inverse_data_process(X)


class EPCA(ComplexPCA):
    """Euler principal component analysis (ePCA).

    Parameter
    ----------
    alpha : flaot or None
        Parameter of euler transform.

    """

    def __init__(self, alpha=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def preprocess(self, X, y=None):
        return self.euler(X)

    def inverse_data_process(self, X, y=None):
        X = self.inverse_euler(X)
        return X

    def euler(self, x):
        z = (cmath.e ** (1j * self.alpha * math.pi * x)) / math.sqrt(2)
        return z

    def inverse_euler(self, z):
        x = (np.angle(z) / (self.alpha * cmath.pi)).real
        return x


class MyEPCA:
    """
    input : (p, a, b)
    output: (m, a, b)
    """

    def __init__(self, n_components=200, alpha=1.9):
        self.n_components = n_components
        self.pca = EPCA(n_components=n_components, alpha=alpha)

    def fit(self, X):
        p, a, b = X.shape
        X = X.reshape(p, a * b).transpose(1, 0)
        self.pca.fit(X)

    def transform(self, X):
        p, a, b = X.shape
        X = X.reshape(p, a * b).transpose(1, 0)
        Y = self.pca.transform(X)
        Y = Y.transpose(1, 0).reshape(self.n_components, a, b)
        return Y
