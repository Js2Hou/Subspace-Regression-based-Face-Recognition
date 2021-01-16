import numpy as np
from sklearn.linear_model import Ridge

from data_loader import DataLoader
from models import ERRClassifier

dataset5_std = DataLoader.load_ExtYaleB(mode='1')


def grid_search():
    train_x, train_y = dataset5_std[0]
    test_x, test_y = dataset5_std[-1]
    lambs = [i / 10 for i in range(1, 11)]
    alphas = [i / 10 - 1 for i in range(20)]

    opt_acc = 0
    opt_paras = (0.5, 0.5)
    for lamb in lambs:
        for alpha in alphas:
            model = ERRClassifier(lamb=lamb, alpha=alpha, m=38)
            model.fit(train_x, train_y)
            acc = model.evaluate(test_x, test_y)
            print(f'lamb:{lamb} alpha:{alpha} acc:{acc}')
            if acc > opt_acc:
                opt_acc = acc
                opt_paras = lamb, alpha

    print(f'optimized parameters: (lamb={opt_paras[0]}, alpha={opt_paras[1]})')


def test():
    train_x, train_y = dataset5_std[0]
    test_x, test_y = dataset5_std[1]
    model = ERRClassifier(m=38)
    model.fit(train_x, train_y)
    acc = model.evaluate(test_x, test_y)
    print(f'acc: {acc}')


# 测试手写的岭回归和sklearn岭回归算出的回归系数的差距
def test2():
    X = np.array([[0, 0], [0, 0], [1, 1]])
    Y = np.array([0, .1, 1])
    alpha = 0.5
    w1 = np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T @ Y
    ridge = Ridge(alpha=alpha)
    ridge.fit(X, Y)
    w2 = ridge.coef_
    diff = np.linalg.norm(w1 - w2)
    print(f'w1 - w2: {diff}')

# if __name__ == '__main__':
#     test2()
