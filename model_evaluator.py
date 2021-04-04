import time
from functools import wraps

from dataset import AR, CUFS, ExtYaleB, Palm
from model.subspace_regression import *


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        func_return = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__}() execute time: {end - start}s')
        return func_return

    return wrapper


@timer
def evaluate_on_AR():
    train_data, test_data = AR.exp2()
    model = ECRC(lamb=600.0, alpha=1.9)
    model.fit(train_data)
    acc = model.score(test_data)
    print(f'{model.__class__.__name__}-AR-exp2: {acc}')


@timer
def evaluate_on_ExtYaleB():
    train_data, *four_test_data = ExtYaleB.load_ExtYaleB_20()
    test_data = four_test_data[-3]
    model = ECRC(lamb=5, alpha=0.9)
    model.fit(train_data)
    acc = model.score(test_data)
    print(f'{model.__class__.__name__} sub5: {acc}')


@timer
def evaluate_on_NUST():
    pass


@timer
def evaluate_on_CUFS():
    train_xs, train_ys, test_xs, test_ys = CUFS.exp1()
    train_xs = train_xs / np.max(train_xs)
    test_xs = test_xs / np.max(test_xs)
    # StandardScaler doesn't work
    # scaler = MinMaxScaler()
    # scaler.fit_transform(train_xs)
    # scaler = MinMaxScaler()
    # scaler.fit_transform(test_xs)

    lambs = [0.5, 1, 5, 10, 50, 100, 500]
    alphas = [0.1, 0.5, 0.9, 1.3, 1.5]
    lamb = 50
    alpha = 0.9
    # model = SRC(lamb=0.5)
    # model.fit(train_xs, train_ys)
    # acc = model.score(test_xs, test_ys)
    # print(f'{model.__class__.__name__}(lamb=0.5) CUFS: {acc}')

    model = ECRC(lamb=lamb, alpha=alpha)
    model.fit(train_xs, train_ys)
    acc = model.score(test_xs, test_ys)
    print(f'{model.__class__.__name__}(lamb={lamb},alpha={alpha}) CUFS: {acc}')

    model = ERRC(lamb=lamb, alpha=alpha)
    model.fit(train_xs, train_ys)
    acc = model.score(test_xs, test_ys)
    print(f'{model.__class__.__name__}(lamb={lamb},alpha={alpha}) CUFS: {acc}')


@timer
def evaluate_on_Palm():
    train_xs, train_ys, *test_data = Palm.exp1(mode=1)
    i = 0  # 第几个测试样本，遮挡比例一次为：0， 0.1， 0.3， 0.5
    test_xs, test_ys = test_data[i * 2: i * 2 + 2]

    lamb = 0.5
    alpha = 1.9

    model = SRC(lamb=lamb)
    model.fit(train_xs, train_ys)
    acc = model.score(test_xs, test_ys)
    print(f'{model.__class__.__name__} Palm(lamb={lamb}, alpha={alpha}): {acc}')

    model = ESRC(lamb=lamb, alpha=alpha)
    model.fit(train_xs, train_ys)
    acc = model.score(test_xs, test_ys)
    print(f'{model.__class__.__name__} Palm(lamb={lamb}, alpha={alpha}): {acc}')
