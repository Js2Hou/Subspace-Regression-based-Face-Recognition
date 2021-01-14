import cmath
import math
import numpy as np


class LRClassifier:
    def __init__(self, train_data):
        """
        :param train_data: shape like (n_features, n_imgs_per_class, n_classes)
        """
        self.train_data = train_data

    def fit(self):
        pass

    def predict(self, test_data):
        """
        :param test_data: shape like (n_features, n_imgs_per_class, n_classes)
        :return:
            pred_y: list
                predicted labels of test dataset
            true_y: list
                true labels of test dataset
        """
        train_data = self.train_data
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
        return pred_y, true_y

    def evaluate(self, test_data):
        """
        :param test_data: shape like (n_features, n_imgs_per_class, n_classes)
        :return:
            acc: double
                classification accuracy on test_data
        """
        y_hat, y = self.predict(test_data)
        n_false = np.count_nonzero(np.array(y_hat) - np.array(y))
        acc = 1 - n_false / len(y)
        return acc


class RRClassifier:
    def __init__(self, m, lamb=0.5):
        self.m = m
        self.lamb = lamb
        self.w = None
        self.T = self.data_prepare(RRClassifier.get_regular_simplex_vertices(self.m))

    def data_prepare(self, X):
        return X

    def fit(self, X, Y):
        X = self.data_prepare(X)
        Y = self.T[Y]
        w = np.linalg.inv(X.T @ X + self.lamb * np.eye(X.shape[1])) @ X.T @ Y
        self.w = w

    def evaluate(self, X, Y):
        X = self.data_prepare(X)
        T = self.T
        n = X.shape[0]
        X_regular = X @ self.w

        count_right = 0
        for x_regular, y_true in zip(X_regular, Y):
            min_distance = 1e10
            y_pred = -1
            for id, t in enumerate(T):
                distance = np.linalg.norm(x_regular - t)
                if distance < min_distance:
                    min_distance = distance
                    y_pred = id
            if y_true == y_pred:
                count_right += 1
        acc = count_right / n
        return acc

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


class ERRClassifier(RRClassifier):
    def __init__(self, alpha, m):
        self.alpha = alpha
        super().__init__(m)

    def data_prepare(self, X):
        return self.euler(X)

    def euler(self, x):
        alpha = self.alpha
        z = (cmath.e ** (1j * alpha * math.pi * x)) / math.sqrt(2)
        return z
