import cmath
import math

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


class LRClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.train_data = None

    def prepare_data(self, X):
        return X

    def fit(self, train_data, y=None):
        self.train_data = train_data
        pass

    def predict(self, test_data, y=None):
        """
        :param y: None
        :param test_data: shape like (n_features, n_imgs_per_class, n_classes)
        :return:
            pred_y: ndarray
                predicted labels of test dataset
            true_y: ndarray
                true labels of test dataset
        """
        train_data = self.prepare_data(self.train_data)
        test_data = self.prepare_data(test_data)

        p, n_per_class, n_class = test_data.shape
        pred_y = []
        true_y = []
        for test_id in range(n_class):
            for ith in range(n_per_class):
                min_dist = 1e10
                id = -1
                y = test_data[:, ith, test_id][:, np.newaxis]
                for train_id in range(n_class):
                    X = train_data[:, :, train_id]
                    y_hat = X @ np.linalg.inv(X.T @ X) @ X.T @ y
                    dist = np.linalg.norm(y - y_hat)
                    if dist < min_dist:
                        min_dist = dist
                        id = train_id
                pred_y.append(id)
                true_y.append(test_id)
        return np.array(pred_y), np.array(true_y)

    def score(self, X, y=None, sample_weight=None):
        """
        :param sample_weight:
        :param y:
        :param X: shape like (n_features, n_imgs_per_class, n_classes)
        :return:
            acc: double
                classification accuracy on test_data
        """
        return accuracy_score(*self.predict(X))


class LRC2(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.train_X = None
        self.train_y = None
        self.n_class = None

    def prepare_data(self, X):
        return X

    def fit(self, X, y):
        X = X.copy()
        X = X[np.argsort(y)]
        y = np.sort(y)
        self.train_X = X
        self.train_y = y
        self.n_class = np.max(y) + 1

    def predict(self, X, y=None):
        X = X.copy()
        train_X = self.prepare_data(self.train_X)
        test_X = self.prepare_data(X)
        train_y = self.train_y

        pred_y = []
        for x_test in test_X:
            x_test = x_test[:, np.newaxis]
            min_dist = 1e10
            id = -1
            for train_id in range(self.n_class):
                X = train_X[train_y == train_id]
                X = X.T
                x_test_hat = self.express_by_dict(X, x_test)
                dist = np.linalg.norm(x_test - x_test_hat)
                if dist < min_dist:
                    min_dist = dist
                    id = train_id
            pred_y.append(id)
        return np.array(pred_y)

    def express_by_dict(self, X, y):
        """
        reconstruct y by X
        :param X: dict
        :param y: sample to be expressed by X
        :return: reconstruct result of y by X
        """
        y_hat = X @ np.linalg.inv(X.T @ X) @ X.T @ y
        return y_hat

    def score(self, X, y, sample_weight=None):
        """
        :param sample_weight:
        :param y:
        :param X: shape like (n_samples, n_features)
        :return:
            acc: classification accuracy on test_data
        """
        y_pred = self.predict(X)
        return accuracy_score(y_pred, y)


class RRC2(LRC2):
    def __init__(self, lamb=0.5):
        super().__init__()
        self.lamb = lamb

    def express_by_dict(self, X, y):
        y_hat = X @ np.linalg.inv(X.T @ X + self.lamb * np.eye(X.shape[1])) @ X.T @ y
        return y_hat


class ERRC2(LRC2):
    def __init__(self, lamb, alpha=0.9):
        super().__init__()
        self.lamb = lamb
        self.alpha = alpha

    def prepare_data(self, X):
        X = self.euler(X)
        return X

    def euler(self, x):
        alpha = self.alpha
        z = (cmath.e ** (1j * alpha * math.pi * x)) / math.sqrt(2)
        return z


class MLRClassifier(LRClassifier):
    def __init__(self, H, W, m=3, n=3, threshold_=None):
        super(MLRClassifier, self).__init__()
        self.W = W
        self.H = H
        self.m = m
        self.n = n
        self.threshold_ = m * n // 3 if threshold_ is None else threshold_

    def prepare_data(self, X):
        """

        :param X: shape like (height, width, n_imgs_per_class, n_classes)
        :return:
            X: shape like (n_features, n_imgs_per_individual, n_chunks, n_classes)
        """
        H, W, m, n = self.H, self.W, self.m, self.n
        n_imgs_per_class, n_classes = X.shape[-2:]
        if len(X.shape) == 3:
            X = X.reshape(H, W, n_imgs_per_class, n_classes)
        h, w = H // m, W // n
        imgs_list = []
        for i in range(m):
            for j in range(n):
                tem = X[i * h:i * h + h, j * w:j * w + w, :, :]
                imgs_list.append(tem.reshape(h * w, n_imgs_per_class, n_classes))
        return np.array(imgs_list).transpose((1, 2, 0, 3))

    def predict(self, test_data, y=None):
        """
        :param y: None
        :param test_data: shape like (height, width, n_imgs_per_class, n_classes)
        :return:
            pred_y: ndarray
                predicted labels of test dataset
            true_y: ndarray
                true labels of test dataset
        """
        train_data = self.prepare_data(self.train_data)
        test_data = self.prepare_data(test_data)

        p, n_per_class, n_chunks, n_class = test_data.shape
        pred_y = []
        true_y = []
        for test_id in range(n_class):
            for ith in range(n_per_class):
                ids = []
                dists = []
                for chunk in range(n_chunks):
                    min_dist = 1e10
                    id = -1
                    y = test_data[:, ith, chunk, test_id][:, np.newaxis]
                    for train_id in range(n_class):
                        X = train_data[:, :, chunk, train_id]
                        y_hat = X @ np.linalg.inv(X.T @ X) @ X.T @ y
                        dist = np.linalg.norm(y - y_hat)
                        if dist < min_dist:
                            min_dist = dist
                            id = train_id
                    ids.append(id)
                    dists.append(min_dist)
                dists, ids = zip(*sorted(zip(dists, ids)))
                ids = ids[:self.threshold_]
                pred_y.append(max(ids, key=ids.count))
                true_y.append(test_id)
        return np.array(pred_y), np.array(true_y)


class RRC2010(LRClassifier):
    def __init__(self, lamb=0.5):
        super().__init__()
        self.lamb = lamb
        pass

    def predict(self, test_data, y=None):
        """
        :param y:
        :param test_data: shape like (n_features, n_imgs_per_class, n_classes)
        :return:
            pred_y: list
                predicted labels of test dataset
            true_y: list
                true labels of test dataset
        """
        train_data = self.prepare_data(self.train_data)
        test_data = self.prepare_data(test_data)
        n_per_class_train = train_data.shape[1]
        p, n_per_class_test, n_class = test_data.shape
        pred_y = []
        true_y = []
        for test_id in range(n_class):
            for ith in range(n_per_class_test):
                min_dist = 1e10
                id = -1
                y = test_data[:, ith, test_id][:, np.newaxis]
                for train_id in range(n_class):
                    X = train_data[:, :, train_id]
                    y_hat = X @ np.linalg.inv(X.T @ X + self.lamb * np.eye(n_per_class_train)) @ X.T @ y
                    dist = np.linalg.norm(y - y_hat)
                    if dist < min_dist:
                        min_dist = dist
                        id = train_id
                pred_y.append(id)
                true_y.append(test_id)
        return np.array(pred_y), np.array(true_y)


class ERRC2010(RRC2010):
    def __init__(self, lamb=0.5, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.lamb = lamb

    def prepare_data(self, X):
        return self.euler(X)

    def predict(self, test_data, y=None):
        """
        :param y:
        :param test_data: shape like (n_features, n_imgs_per_class, n_classes)
        :return:
            pred_y: list
                predicted labels of test dataset
            true_y: list
                true labels of test dataset
        """
        train_data = self.prepare_data(self.train_data)
        test_data = self.prepare_data(test_data)
        n_per_class_train = train_data.shape[1]
        p, n_per_class_test, n_class = test_data.shape
        pred_y = []
        true_y = []
        for test_id in range(n_class):
            for ith in range(n_per_class_test):
                min_dist = 1e10
                id = -1
                y = test_data[:, ith, test_id][:, np.newaxis]
                for train_id in range(n_class):
                    X = train_data[:, :, train_id]
                    y_hat = X @ np.linalg.inv(np.conj(X).T @ X + self.lamb * np.eye(n_per_class_train)) @ np.conj(
                        X).T @ y
                    dist = np.linalg.norm(y - y_hat)
                    if dist < min_dist:
                        min_dist = dist
                        id = train_id
                pred_y.append(id)
                true_y.append(test_id)
        return np.array(pred_y), np.array(true_y)

    def euler(self, x):
        alpha = self.alpha
        z = (cmath.e ** (1j * alpha * math.pi * x)) / math.sqrt(2)
        return z


class ERRC2010Real(RRC2010):
    def __init__(self, lamb=0.5, alpha=0.9, option='real'):
        super().__init__()
        self.alpha = alpha
        self.lamb = lamb
        self.option = option

    def prepare_data(self, X):
        mdict = ['real', 'imag', 'abs']
        if self.option not in mdict:
            print(f'there is no option named {self.option}!')
            return

        if self.option is 'real':
            return np.real(self.euler(X))
        if self.option is 'imag':
            return np.imag(self.euler(X))
        if self.option is 'abs':
            return np.abs(self.euler(X))

    def euler(self, x):
        alpha = self.alpha
        z = (cmath.e ** (1j * alpha * math.pi * x)) / math.sqrt(2)
        return z


class RRC2007(BaseEstimator, ClassifierMixin):
    def __init__(self, m, lamb=0.5):
        self.m = m
        self.lamb = lamb
        self.coeff_ = None
        self.T = self.prepare_data(RRC2007.get_regular_simplex_vertices(self.m))

    def prepare_data(self, X):
        return X

    def fit(self, X, Y):
        X = self.prepare_data(X)
        Yt = self.T[Y]
        self.coeff_ = np.linalg.inv(X.T @ X + self.lamb * np.eye(X.shape[1])) @ X.T @ Yt
        return self

    def predict(self, X):
        X = self.prepare_data(X)
        T = self.T
        Yt_hat = X @ self.coeff_
        Y_hat = []
        for yt_hat in Yt_hat:
            min_distance = 1e10
            y_pred = -1
            for id, t in enumerate(T):
                distance = np.linalg.norm(yt_hat - t)
                if distance < min_distance:
                    min_distance = distance
                    y_pred = id
            Y_hat.append(y_pred)
        return np.array(Y_hat)

    @staticmethod
    def get_regular_simplex_vertices(m):
        T = np.zeros((m - 1, m))
        T[0, 0] = 1
        for i in range(1, m):
            T[0, i] = 1 / (m - i)
        for i in range(m - 1):
            T[i][i] = math.sqrt(1 - np.sum(T[:i, i] * T[:i, i]))
            for j in range(i + 1, m):
                T[i, j] = -T[i, i] / (m - i - 1)
        T = T.T
        return T


class ERRC2007(RRC2007):
    def __init__(self, m, lamb=0.5, alpha=0.9):
        self.alpha = alpha
        super().__init__(lamb=lamb, m=m)

    def prepare_data(self, X):
        return self.euler(X)

    def euler(self, x):
        alpha = self.alpha
        z = (cmath.e ** (1j * alpha * math.pi * x)) / math.sqrt(2)
        return z


class ComplexPCA(object):
    """Complex principle component analysis

    Input
    --------
    X : array-like of shape (n_samples, n_features)
        X is a complex-valued matrix

    Parameter
    --------
    n_components : int or None
        Numbers of components to keep.
        if n_components is not set, then all components are kept:
            n_components == min(n_samples, n_features)

    threshold : float or None
        Lower bound constraint on the sum of the explained variance ratios of components. If n_components is specified,
        this parameter is ignored.

    Attributes
    --------
    components_ : array, shape (n_components, n_features)
        Principal axes in feature space, representing the directions of maximum variance in the data. The components
        are sorted by ``explained_variance_``.

    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each of the selected components.

        If ``n_components`` and ``total`` are not set then all components are stored and the sum of the ratios is equal
        to 1.0.

    n_components_ : int
        The estimated number of components. If n_components is not specified, program will auto compute n_components_
        by the given explained_variance_ratio_sum. Otherwise, it equals the parameter n_components.

    n_features_ : int
        Number of features in the training data.

    n_samples_ : int
        Number of samples in the training data.
    """

    def __init__(self, n_components=None, threshold=None):
        self.n_components = n_components
        self.threshold = threshold
        self.mmscale = MinMaxScaler()

    def data_process(self, X, y=None):
        # return self.mmscale.transform(X)
        return X

    def inverse_data_process(self, X, y=None):
        # return self.mmscale.inverse_transform(X)
        return X

    def fit(self, X, y=None):
        """Fit the model with X.

        Parameters
        --------
        X : array_like, shape (n_samples, n_features)
            Training data, where n_samples if the number of samples and n_features if the number of features

        y : None
            Ignored variable

        Returns
        --------
        self : object
            Returns the instance itself.
        """
        X = X.copy()
        self.mmscale.fit(X)
        self._fit(X)
        return self

    def _fit(self, X):
        """Fit the model by computing eigenvalue decomposition on X * X.H"""
        Z = self.data_process(X)

        n_samples, n_features = Z.shape

        # Handle n_components==None
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
        Z = self.data_process(Z)
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
        self.mmscale = MinMaxScaler()

    def data_process(self, X, y=None):
        # X = self.mmscale.transform(X)
        return self.euler(X)

    def inverse_data_process(self, X, y=None):
        X = self.inverse_euler(X)
        # return self.mmscale.inverse_transform(X)
        return X

    def euler(self, x):
        z = (cmath.e ** (1j * self.alpha * math.pi * x)) / math.sqrt(2)
        return z

    def inverse_euler(self, z):
        x = (np.angle(z) / (self.alpha * cmath.pi)).real
        return x  # return (-1j * np.log(np.sqrt(2) * z)).real


class MyPCA:
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
