import time

from data_loader import DataLoader
from models import *


def timer(func):
    # 定义嵌套函数，用来打印出装饰的函数的执行时间
    def wrapper(*args, **kwargs):
        # 定义开始时间和结束时间，将func夹在中间执行，取得其返回值
        start = time.time()
        func_return = func(*args, **kwargs)
        end = time.time()
        # 打印方法名称和其执行时间
        print(f'{func.__name__}() execute time: {end - start}s')
        # 返回func的返回值
        return func_return

    # 返回嵌套的内层函数
    return wrapper


@timer
def evaluate_on_AR():
    # 导入数据
    train_data, test_data = DataLoader.load_AR(mode='2')
    train_x, train_y, test_x, test_y = DataLoader.load_AR(mode='1')

    # 线性回归分类器
    model1 = LRClassifier(train_data)
    acc = model1.evaluate(test_data)

    with open(r'./results/a.txt', 'a+') as f:
        cur_time = time.strftime("%Y-%m-%d %X", time.localtime())
        f.write(f'{cur_time} LRC-AR: {acc}\n')

    # 岭回归分类器
    model2 = RRClassifier(m=120)
    model2.fit(train_x, train_y)
    acc2 = model2.evaluate(test_x, test_y)

    with open(r'./results/a.txt', 'a+') as f:
        cur_time = time.strftime("%Y-%m-%d %X", time.localtime())
        f.write(f'{cur_time} RRC-AR: {acc2}\n')

    # 欧拉岭回归分类器
    model3 = ERRClassifier(alpha=0.9, m=120)
    model3.fit(train_x, train_y)
    acc3 = model3.evaluate(test_x, test_y)

    with open(r'./results/a.txt', 'a+') as f:
        cur_time = time.strftime("%Y-%m-%d %X", time.localtime())
        f.write(f'{cur_time} ERRC-AR: {acc3}\n')


@timer
def evaluate_on_ExtYaleB():
    dataset_xy = DataLoader.load_ExtYaleB(mode=1)
    train_x, train_y = dataset_xy[0]

    dataset = DataLoader.load_ExtYaleB(mode=2)
    # 线性回归分类器
    model1 = LRClassifier(dataset[0])
    for i in range(1, 5):
        acc = model1.evaluate(dataset[i])
        with open(r'./results/a.txt', 'a+') as f:
            cur_time = time.strftime("%Y-%m-%d %X", time.localtime())
            f.write(f'{cur_time} LRC-ExtYaleB(condition {i}): {acc}\n')

    # 岭回归分类器
    model2 = RRClassifier(m=38)
    model2.fit(train_x, train_y)
    for i in range(1, 5):
        test_x, test_y = dataset_xy[i]
        acc2 = model2.evaluate(test_x, test_y)
        with open(r'./results/a.txt', 'a+') as f:
            cur_time = time.strftime("%Y-%m-%d %X", time.localtime())
            f.write(f'{cur_time} RRC-ExtYaleB(condition {i}): {acc2}\n')

    # 欧拉岭回归分类器
    model3 = ERRClassifier(alpha=0.9, m=38)
    model3.fit(train_x, train_y)
    for i in range(1, 5):
        test_x, test_y = dataset_xy[i]
        acc3 = model3.evaluate(test_x, test_y)
        with open(r'./results/a.txt', 'a+') as f:
            cur_time = time.strftime("%Y-%m-%d %X", time.localtime())
            f.write(f'{cur_time} ERRC-ExtYaleB(condition {i}): {acc3}\n')
