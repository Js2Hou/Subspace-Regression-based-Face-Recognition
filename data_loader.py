import os
import glob

import numpy as np
from scipy.io import loadmat
from sklearn.utils import shuffle


class DataLoader:
    def __init__(self):
        pass

    @staticmethod
    def load_AR(path=r'./data/AR_120_26_50_40.mat', mode='1'):
        """ Provide two mode load dataset AR.

        :param path: path of data
        :param mode: '1' or '2'
            '1' represents standard mode, while '2' represents integration mode.
        :return:
            mode '1':
                [train_x, train_y, test_x, test_y]
                train_x's shape like (n_samples, n_features)
            mode '2':
                [train_data, test_data]
                train_data's shape like (n_features, n_img_per_class, n_classes)
        """

        # 导入数据
        data = loadmat(path)['DAT']  # (2000, 26, 120)
        # 划分训练集和测试集
        train_data, test_data = data[:, :13, :], data[:, :13, :]
        if mode == '1' or mode == 1:
            # 制作训练集和测试集：(n_samples, n_features)
            train_x = np.vstack([train_data[:, i, j] for j in range(120) for i in range(13)])
            train_y = np.tile(np.arange(120), 13)
            test_x = np.vstack([test_data[:, i, j] for j in range(120) for i in range(13)])
            test_y = train_y.copy()
            # 打乱训练集和测试集顺序
            train_x, train_y = shuffle(train_x, train_y)
            test_x, test_y = shuffle(test_x, test_y)
            # 返回训练集和测试集
            return train_x, train_y, test_x, test_y
        elif mode == '2' or mode == 2:
            return train_data, test_data
        else:
            print(f'there is no mode named {mode}!')

    @staticmethod
    def load_ExtYaleB(path=r'./data/ExtYaleB_illumination/', mode='1'):
        """

        :param path: path of .mat file
        :param mode: '1' or '2'
            '1' represents standard mode, while '2' represents integration mode.
        :return:
            mode '1':
                [(train_x, train_y), (test1_x, test1_y), ..., (test4_x, test4_y)]
                shape like: (n_samples, n_features)
            mode '2':
                [train_data, test1_data, ..., test4_data]
                shape like: (n_features, n_imgs_per_class, n_classes)
        """

        data5 = []
        path5 = os.listdir(path)
        path5.sort()  # sorted by name

        for file in path5:
            data = loadmat(os.path.join(path, file))['DAT']
            data5.append(data)

        result = []
        if mode == '1' or mode == 1:  # 标准模式
            for dataset in data5:
                _, n_imgs_per_class, n_classes = dataset.shape
                data_x = np.vstack([dataset[:, i, j] for j in range(n_classes) for i in range(n_imgs_per_class)])
                data_y = np.tile(np.arange(n_classes), n_imgs_per_class)
                data_x, data_y = shuffle(data_x, data_y)
                result.append((data_x, data_y))
            return result
        elif mode == '2' or mode == 2:  # 集成模式
            return data5
        else:
            print(f'there is no mode named {mode}!')


if __name__ == '__main__':
    print('start')
    dataset = DataLoader.load_ExtYaleB(mode='1')
    for data in dataset:
        print(f'x: {data[0].shape} y: {data[1].shape}')
