# -*- coding: utf-8 -*-
"""
@Time ： 2021/3/25 10:44
@Auth ： JsHou
@Email : 137073284@qq.com
@File ：_base.py

"""
import math
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.metrics import accuracy_score


class _Base(metaclass=ABCMeta):

    @staticmethod
    def _make_shape(X):
        _, a_, b_ = X.shape
        xs = np.vstack([X[:, i, j] for i in range(a_) for j in range(b_)])
        ys = np.tile(np.arange(b_), a_)
        return xs, ys

    @abstractmethod
    def preprocess(self, X):
        return X

    def fit(self, X, y=None):
        if y is None:
            X, y = self._make_shape(X)
        X = self.preprocess(X)
        self._fit(X, y)

    @abstractmethod
    def _fit(self, X, y):
        pass

    def predict(self, X):
        if len(X.shape) == 3:
            X, _ = self._make_shape(X)
        X = self.preprocess(X)
        return self._predict(X)

    @abstractmethod
    def _predict(self, X):
        pass

    def score(self, X, y=None):
        if y is None:
            X, y = self._make_shape(X)
        pred_y = self.predict(X)
        return accuracy_score(pred_y, y)


class _CRC(_Base):
    """
    Collaborative Representation Classifier.
    """

    def preprocess(self, X):
        return X

    @abstractmethod
    def _fit(self, X, y):
        self.X_train = X
        self.Y_train = y

    @abstractmethod
    def get_coeff(self, X, y, *args):
        pass

    @staticmethod
    def get_residual_error(X, y, coeff_):
        return np.linalg.norm(coeff_ @ X - y) / np.linalg.norm(coeff_)

    def _predict(self, X):
        # todo: 标签不连续或者不是从0开始时，这样处理是有问题的
        num_classes = np.max(self.Y_train) + 1
        Y_pred = []
        for x in X:
            min_residual_error = 1e10
            pred_y = -1
            x = x.reshape(1, -1)
            coeff = self.get_coeff(self.X_train, x)  # 自定义
            for i in range(num_classes):
                index = (self.Y_train == i)
                coeff_ = coeff[:, index]
                X_ = self.X_train[index, :]  # 训练集中第i类全体样本，(n_i, p)
                residual_error_ = self.get_residual_error(X_, x, coeff_)  # 自定义
                if min_residual_error > residual_error_:
                    min_residual_error = residual_error_
                    pred_y = i
            Y_pred.append(pred_y)
        return np.array(Y_pred)


class _LRC(_Base):
    """
    Linear Regression Classifier.
    """

    def preprocess(self, X):
        return X

    @abstractmethod
    def _fit(self, X, y):
        self.X_train = X
        self.Y_train = y

    @abstractmethod
    def get_coeff(self, X, y, *args, **kwargs):
        pass

    @staticmethod
    def get_residual_error(X, y, coeff_):
        # return np.linalg.norm(coeff_ @ X - y) / np.linalg.norm(coeff_)
        return np.linalg.norm(coeff_ @ X - y)

    def _predict(self, X):
        # todo: 标签不连续或者不是从0开始时，这样处理是有问题的
        num_classes = np.max(self.Y_train) + 1
        Y_pred = []
        for x in X:
            x = x.reshape(1, -1)
            min_residual_error = 1e10
            pred_y = -1
            for i in range(num_classes):
                index = (self.Y_train == i)
                X_ = self.X_train[index, :]  # 训练集中第i类全体样本，(n_i, p)
                coeff_ = self.get_coeff(X_, x, **{'class_id': i})
                residual_error_ = self.get_residual_error(X_, x, coeff_)  # 自定义
                if min_residual_error > residual_error_:
                    min_residual_error = residual_error_
                    pred_y = i
            Y_pred.append(pred_y)
        return np.array(Y_pred)


class _Euler:
    @staticmethod
    def euler(x, alpha=1.9):
        z = np.exp(1j * alpha * math.pi * x) / math.sqrt(2)
        # z = (cmath.e ** (1j * alpha * math.pi * x)) / math.sqrt(2)
        return z
