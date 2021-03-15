import time

from models import *
from utils import *


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
    # experiment1
    train_data, test_data = Dataloader.load_ar_exp1()
    train_data2, test_data2 = Dataloader.load_ar_exp2()
    train_data3, test_data3 = Dataloader.load_ar_exp3()

    model = LRClassifier()
    model.fit(train_data)
    acc = model.score(test_data)
    model.fit(train_data2)
    acc2 = model.score(test_data2)
    model.fit(train_data3)
    acc3 = model.score(test_data3)
    print(f'LRC  exp1:{acc} exp2:{acc2} exp3:{acc3}')

    model = RRC2010()
    model.fit(train_data)
    acc = model.score(test_data)
    model.fit(train_data2)
    acc2 = model.score(test_data2)
    model.fit(train_data3)
    acc3 = model.score(test_data3)
    print(f'RRC  exp1:{acc} exp2:{acc2} exp3:{acc3}')

    model = ERRC2010(alpha=1.9)
    model.fit(train_data)
    acc = model.score(test_data)
    model.fit(train_data2)
    acc2 = model.score(test_data2)
    model.fit(train_data3)
    acc3 = model.score(test_data3)
    print(f'ERRC(lamb=0.5)  exp1:{acc} exp2:{acc2} exp3:{acc3}')

    model = ERRC2010(lamb=0, alpha=1.9)
    model.fit(train_data)
    acc = model.score(test_data)
    model.fit(train_data2)
    acc2 = model.score(test_data2)
    model.fit(train_data3)
    acc3 = model.score(test_data3)
    print(f'ERRC(lamb=0)  exp1:{acc} exp2:{acc2} exp3:{acc3}')


@timer
def evaluate_on_ExtYaleB():
    """ based on LRC-TPAMI2010

    :return:
    """
    dataset = Dataloader.load_ExtYaleB_cropped()
    data1 = dataset[0]
    data2 = dataset[-1]
    data = np.concatenate((data1, data2), axis=1)
    train_data = data[:, :8, :]
    test_data0 = data[:, 8:, :]

    lamb = 0.01
    # alphas = [i / 10 for i in range(1, 20)]
    alpha = 0.2

    for i in range(1, 5):
        print(f'子数据集{i}')
        test_data = test_data0 if i is 5 else dataset[i]
        # 线性回归分类器
        model1 = LRClassifier()
        model1.fit(train_data)
        acc = model1.score(test_data)
        print(f'LRC acc: {acc}')

        # 岭回归分类器
        model2 = RRC2010(lamb=lamb)
        model2.fit(train_data)
        acc2 = model2.score(test_data)
        print(f'RRC acc: {acc2}')

        # 欧拉岭回归分类器
        model3 = ERRC2010(lamb=lamb, alpha=alpha)
        model3.fit(train_data)
        acc3 = model3.score(test_data)
        print(f'alpha={alpha} lamb={lamb} acc={acc3}')


@timer
def evaluate_on_ExtYaleB2():
    dataset = Dataloader.load_ExtYaleB_cropped()
    data = np.concatenate((dataset[0], dataset[1], dataset[2], dataset[3], dataset[4]), axis=1)
    #  7  12  12  14  19
    # 0  7  19  31  45  64
    train_data = data[:, [*range(8), 19, 31, 45], :]
    dataset = [data[:, 8:19, :], data[:, 20:31, :], data[:, 32:45, :], data[:, 46:, :]]

    lamb = 0.01
    # alphas = [i / 10 for i in range(1, 20)]
    alpha = 0.2

    for i in range(4):
        print(f'子数据集{i + 2}')
        test_data = dataset[i]
        # 线性回归分类器
        model1 = LRClassifier()
        model1.fit(train_data)
        acc = model1.score(test_data)
        print(f'LRC acc: {acc}')

        # 岭回归分类器
        # model2 = RRC2010(lamb=lamb)
        # model2.fit(train_data)
        # acc2 = model2.score(test_data)
        # print(f'RRC acc: {acc2}')

        # 欧拉岭回归分类器
        model3 = ERRC2010(lamb=0, alpha=alpha)
        model3.fit(train_data)
        acc3 = model3.score(test_data)
        print(f'alpha={alpha} lamb={lamb} acc={acc3}')


@timer
def evaluate_ERRC_on_ExtYaleB_noised():
    train_x, test_x, train_y, test_y = Dataloader.load_ExtYaleB_cropped2(test_size=0.3, load_ratio=1)

    SNRs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    SNRs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    alphas = [0.4]
    # alphas = [i / 10 for i in range(1, 20)]
    accs = []
    acc2s = []
    for snr in SNRs:
        print(f'SNR={snr}')
        test_x2 = batch_addsalt_pepper(test_x, SNR=snr)
        # img = test_x2[0].reshape(20, 20).T
        # plt.imsave(f'./results/snr-{snr}.jpg', img)

        lamb = 0.5

        rrc = RRC2(lamb=lamb)
        rrc.fit(train_x, train_y)
        acc = rrc.score(test_x2, test_y)
        accs.append(acc)
        print(f'acc for rrc on extyaleb: {acc}')

        for alpha in alphas:
            errc = ERRC2(lamb=lamb, alpha=alpha)
            errc.fit(train_x, train_y)
            acc2 = errc.score(test_x2, test_y)
            acc2s.append(acc2)
            # if acc2 > acc:
            print(f'acc for errc on extyaleb(alpha={alpha}): {acc2}')
    print(f'rrc: {accs}')
    print(f'errc: {acc2s}')
