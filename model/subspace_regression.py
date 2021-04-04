# -*- coding: utf-8 -*-
"""
@Time ： 2021/3/25 11:20
@Auth ： JsHou
@Email : 137073284@qq.com
@File ：subspace_regression.py

"""

import cvxpy as cp
import numpy as np

from model._base import _CRC, _Euler, _LRC


# test ok
class CRC(_CRC):
    def __init__(self, lamb=0.5):
        self.lamb = lamb
        self.P = None  # ndarray, shape like (p, n)
        self.X_train = None
        self.Y_train = None

    def preprocess(self, X):
        return X

    def _fit(self, X, y):
        self.P = X.T @ np.linalg.inv(X @ X.T + self.lamb * np.eye(X.shape[0]))
        self.X_train = X
        self.Y_train = y

    def get_coeff(self, X, y, *args):
        return y @ self.P


# test ok
class ECRC(CRC, _Euler):
    def __init__(self, lamb, alpha):
        super(ECRC, self).__init__(lamb)
        self.alpha = alpha

    def _fit(self, X, y):
        self.P = np.conj(X.T) @ np.linalg.inv(X @ np.conj(X.T) + self.lamb * np.eye(X.shape[0]))
        self.X_train = X
        self.Y_train = y

    def preprocess(self, X):
        return self.euler(X, self.alpha)


# test ok
class SRC(_CRC):
    def __init__(self, lamb=0.5):
        self.lamb = lamb
        self.X_train = None
        self.Y_train = None

    def preprocess(self, X):
        return X

    def _fit(self, X, y):
        self.X_train = X
        self.Y_train = y

    def get_coeff(self, X, y, *args):
        lamb = args[0] if args else 0
        x = cp.Variable((1, X.shape[0]))
        objective_function = cp.Minimize(lamb * cp.norm1(x) + cp.norm2(x @ X - y))
        constraints = []
        prob = cp.Problem(objective_function, constraints)
        prob.solve(solver=cp.SCS)
        # print(prob.status)
        assert str(prob.status) == 'optimal', 'solve regression problem failed'
        return x.value


class ESRC(SRC, _Euler):
    def __init__(self, lamb, alpha):
        super().__init__(lamb)
        self.alpha = alpha

    def preprocess(self, X):
        return self.euler(X)


# test ok
class LRC(_LRC):
    def __init__(self):
        self.X_train = None
        self.Y_train = None
        self.P = None

    def _fit(self, X, y):
        self.X_train = X
        self.Y_train = y
        self.calculate_project_matrix(X, y)

    def calculate_project_matrix(self, X, y):
        num_classes = np.max(y) + 1
        P = []
        for i in range(num_classes):
            X_ = X[y == i]
            P_ = X_.T @ np.linalg.inv(X_ @ X_.T)  # (p, n_i)
            P.append(P_)
        self.P = P

    def get_coeff(self, X, y, *args, **kwargs):
        class_id = kwargs['class_id']
        return y @ self.P[class_id]


# test ok
class RRC(LRC):
    def __init__(self, lamb=0.5):
        super(RRC, self).__init__()
        self.lamb = lamb

    def calculate_project_matrix(self, X, y):
        num_classes = np.max(y) + 1
        P = []
        for i in range(num_classes):
            X_ = X[y == i]
            n_i = X_.shape[0]
            P_ = X_.T @ np.linalg.inv(X_ @ X_.T + self.lamb * np.eye(n_i))
            P.append(P_)
        self.P = P


# test ok
class ERRC(RRC, _Euler):
    def __init__(self, lamb, alpha):
        super().__init__(lamb)
        self.alpha = alpha

    def preprocess(self, X):
        # return X
        return self.euler(X, self.alpha)

    def calculate_project_matrix(self, X, y):
        num_classes = np.max(y) + 1
        P = []
        for i in range(num_classes):
            X_ = X[y == i]
            n_i = X_.shape[0]
            P_ = np.conj(X_).T @ np.linalg.inv(X_ @ np.conj(X_).T + self.lamb * np.eye(n_i))  # (p, n_i)
            P.append(P_)
        self.P = P
