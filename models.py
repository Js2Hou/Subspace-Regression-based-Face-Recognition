import cmath
import math

import numpy as np

from data_loader import DataLoader


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
        self.coeff_ = None
        self.T = self.prepare_data(RRClassifier.get_regular_simplex_vertices(self.m))

    def prepare_data(self, X):
        return X

    def fit(self, X, Y):
        X = self.prepare_data(X)
        Yt = self.T[Y]
        self.coeff_ = np.linalg.inv(X.T @ X + self.lamb * np.eye(X.shape[1])) @ X.T @ Yt

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

    def evaluate(self, X, Y):
        Y_hat = self.predict(X)
        n_false = np.count_nonzero(Y - Y_hat)
        acc = 1 - n_false / max(Y.shape)
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
    def __init__(self, m, lamb=0.5, alpha=0.9):
        self.alpha = alpha
        super().__init__(lamb=lamb, m=m)

    def prepare_data(self, X):
        return self.euler(X)

    def euler(self, x):
        alpha = self.alpha
        z = (cmath.e ** (1j * alpha * math.pi * x)) / math.sqrt(2)
        return z


if __name__ == '__main__':
    # 导入数据
    train_x, train_y, test_x, test_y = DataLoader.load_AR(mode='1')
    lamb = 0.5
    model = RRClassifier(lamb=lamb, m=120)
    model.fit(train_x, train_y)
    train_y_hat = model.predict(train_x)
    test_y_hat = model.predict(test_x)
    print(train_y_hat[:20])
    print(test_y_hat[:20])
    print(f'acc: {model.evaluate(train_x, train_y)}')
