import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from data_loader import *
from models import *
from utils import *


# 测试ERRC_based_LRC基于ExtYaleB数据集
def test_errc2010_ext():
    dataset5 = Dataloader.load_ExtYaleB_cropped()
    train_data = dataset5[0]
    test_data = dataset5[-1]
    # alphas = [i / 10 for i in range(20)]
    # for alpha in alphas:
    lambs = [i / 100 for i in range(10)]
    for lamb in lambs:
        model = ERRC2010(lamb=lamb, alpha=0.9)
        model.fit(train_data)
        acc = model.score(test_data)
        print(f'lamb: {lamb} acc: {acc}')


# ERRC2 on extyaleb: 超参数选择lamb
def select_lamb_errc2_ext():
    train_x, test_x, train_y, test_y = Dataloader.load_ExtYaleB_cropped2(test_size=0.2, load_ratio=0.5)
    lambs1 = [i / 10 for i in range(1, 10)]
    lambs2 = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    lambs3 = [20, 30, 40, 50, 100]
    lambs = lambs1 + lambs2 + lambs3

    rrc = RRC2()
    param_grid = dict(lamb=lambs)  # 转化为字典格式，网络搜索要求
    kflod = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    clf = GridSearchCV(rrc, param_grid, scoring='accuracy', n_jobs=-1, cv=kflod)
    clf.fit(train_x, train_y)
    print(f'best para: {clf.best_params_}, best score: {clf.best_score_}')


# ERRC2 on extyaleb: 超参数选择alpha
def select_alpha_errc2_ext():
    train_x, test_x, train_y, test_y = Dataloader.load_ExtYaleB_cropped2(test_size=0.2, load_ratio=0.5)
    test_x = batch_addsalt_pepper(test_x, SNR=0.5)
    alphas = [i / 10 for i in range(1, 20)]

    errc = ERRC2(lamb=0.5)
    param_grid = dict(alpha=alphas)  # 转化为字典格式，网络搜索要求
    kflod = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    clf = GridSearchCV(errc, param_grid, scoring='accuracy', n_jobs=-1, cv=kflod)
    clf.fit(train_x, train_y)
    print(f'best para: {clf.best_params_}, best score: {clf.best_score_}')


# 测试手写的岭回归和sklearn岭回归算出的回归系数的差距
def test_diff_ridge_betw_sklearn_mine():
    X = np.array([[0, 0], [0, 0], [1, 1]])
    Y = np.array([0, .1, 1])
    alpha = 0.5
    w1 = np.linalg.inv(X.T @ X + alpha * np.eye(X.shape[1])) @ X.T @ Y
    ridge = Ridge(alpha=alpha)
    ridge.fit(X, Y)
    w2 = ridge.coef_
    diff = np.linalg.norm(w1 - w2)
    print(f'w1 - w2: {diff}')


# 椒盐噪声绘图
def plot_pepper():
    train_x, test_x, train_y, test_y = Dataloader.load_ExtYaleB_cropped2(test_size=0.3)

    img0 = test_x[0].reshape(20, 20).T
    img1 = test_x[1].reshape(20, 20).T
    img2 = test_x[2].reshape(20, 20).T

    snrs = [1, 0.8, 0.6, 0.4, 0.2]
    plt.figure()
    for id, snr in enumerate(snrs):
        img0_ = addsalt_pepper(img0, snr)
        img1_ = addsalt_pepper(img1, snr)
        img2_ = addsalt_pepper(img2, snr)
        plt.subplot(3, 5, id + 1)
        plt.imshow(img0_, cmap='gray')
        plt.title(f'{snr}')
        plt.axis('off')

        plt.subplot(3, 5, id + 6)
        plt.imshow(img1_, cmap='gray')
        plt.axis('off')

        plt.subplot(3, 5, id + 11)
        plt.imshow(img2_, cmap='gray')
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    plot_pepper()
