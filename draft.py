import time

from sklearn.linear_model import Ridge

from models import *


# model based LRC
def grid_search2():
    dataset5 = DataLoader.load_ExtYaleB(mode='2')
    train_data, test_data = dataset5[0], dataset5[-1]
    lambs = [i / 10 for i in range(1, 11)]
    alphas = [i / 10 - 1 for i in range(20)]

    opt_acc = 0
    opt_paras = (0.5, 0.5)
    for lamb in lambs:
        for alpha in alphas:
            clf = ERRC2(train_data, lamb=lamb, alpha=alpha)
            acc = clf.evaluate(test_data)
            cur_time = time.strftime("%Y-%m-%d %X", time.localtime())
            string = f'{cur_time} LRC based lamb:{lamb} alpha:{alpha} acc:{acc}'
            with open(r'./results/a.txt', 'a+') as f:
                f.write(string)
            print(string)
            if acc > opt_acc:
                opt_acc = acc
                opt_paras = lamb, alpha

    print(f'optimized parameters on ExtYaleB: (lamb={opt_paras[0]}, alpha={opt_paras[1]})')


def grid_search():
    dataset5_std = DataLoader.load_ExtYaleB(mode='1')
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

    print(f'optimized parameters on ExtYaleB: (lamb={opt_paras[0]}, alpha={opt_paras[1]})')


# 测试LRC在ExtYaleB上的性能
def test():
    dataset5_std = DataLoader.load_ExtYaleB(mode='2')
    train_data = dataset5_std[0][:, :, :8]
    test_data = dataset5_std[1][:, :, :8]
    model = LRClassifier(train_data=train_data)
    acc = model.evaluate(test_data)
    print(f'acc: {acc}')


# 测试LRC在AR上的性能
def test2():
    # 导入数据
    train_data, test_data = DataLoader.load_AR(mode='2')
    train_data = train_data[:, :, :8]
    test_data = test_data[:, :, :8]
    model = LRClassifier(train_data=train_data)
    acc = model.evaluate(test_data)
    print(f'acc: {acc}')
    cur_time = time.strftime("%Y-%m-%d %X", time.localtime())
    with open(r'./results/a.txt', 'a+') as f:
        f.write(f'{cur_time} LRC-AR: {acc}\n')


# 测试RRC在AR上的性能
def test4():
    # 导入数据
    train_data, test_data = DataLoader.load_AR(mode='2')
    train_data = train_data[:, :, :8]
    test_data = test_data[:, :, :8]
    model = LRClassifier(train_data=train_data)
    acc = model.evaluate(test_data)
    print(f'acc: {acc}')
    cur_time = time.strftime("%Y-%m-%d %X", time.localtime())
    with open(r'./results/a.txt', 'a+') as f:
        f.write(f'{cur_time} LRC-AR: {acc}\n')


# 测试手写的岭回归和sklearn岭回归算出的回归系数的差距
def test3():
    X = np.array([[0, 0], [0, 0], [1, 1]])
    Y = np.array([0, .1, 1])
    alpha = 0.5
    w1 = np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T @ Y
    ridge = Ridge(alpha=alpha)
    ridge.fit(X, Y)
    w2 = ridge.coef_
    diff = np.linalg.norm(w1 - w2)
    print(f'w1 - w2: {diff}')


if __name__ == '__main__':
    test2()
