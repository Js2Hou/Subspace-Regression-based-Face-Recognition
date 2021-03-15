import os

import cv2
import numpy as np
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class Dataloader:
    def __init__(self):
        pass

    @staticmethod
    def load_ExtYaleB_base(path=r'./data/ExtYaleB_illumination/', mode='1'):
        """

        :param img_size: size of images for training
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
                data_x = np.vstack([dataset[:, i, j] for i in range(n_imgs_per_class) for j in range(n_classes)])
                data_y = np.tile(np.arange(n_classes), n_imgs_per_class)
                data_x, data_y = shuffle(data_x, data_y)
                result.append((data_x, data_y))
            return result
        elif mode == '2' or mode == 2:  # 集成模式
            return data5
        else:
            print(f'there is no mode named {mode}!')

    @staticmethod
    def load_ExtYaleB_cropped(path=r'./data/ExtYaleB_20_20.mat'):
        """

        :param path: path of .mat file
        :return:
                [train_data, test1_data, ..., test4_data]
                shape like: (n_features, n_imgs_per_class, n_classes)
        """

        dataset5_dict = loadmat(path)
        data5 = [dataset5_dict[f'sub{i}'] for i in range(1, 6)]
        return data5

    @staticmethod
    def load_ExtYaleB_cropped2(test_size=0.2, load_ratio=0.5):
        """
        前3份数据集混合在一起，测试集上加高斯或椒盐噪声
        :param load_ratio: float
            载入数据的比例
        :param test_size: float
            测试集比例
        :return: train_x, test_x, train_y, test_y
        """
        dataset = Dataloader.load_ExtYaleB_cropped()
        data = np.concatenate((dataset[0], dataset[1], dataset[2]), axis=1)
        p, a, b = data.shape
        data_x = np.vstack([data[:, i, j] for i in range(a) for j in range(b)])
        data_y = np.tile(np.arange(b), a)
        # scaler = MinMaxScaler()
        # scaler.fit_transform(data_x)

        # 数据载入比例，用于小样本超参数选择
        data_x, _, data_y, _ = train_test_split(data_x, data_y, test_size=load_ratio)
        return train_test_split(data_x, data_y, test_size=test_size)

    @staticmethod
    def load_ar(path=r'./data/AR_120_26_20_20.mat'):
        data = loadmat(path)['DAT']  # shape like (p, n_imgs, n_classes)
        return data / np.max(data)

    @staticmethod
    def load_ar_exp1():
        """
        Training sample of sunglass occlusion.
        :return:
        """
        data = Dataloader.load_ar()
        train_data = data[:, :8, :]
        test_data = data[:, [8, 9, *(range(13, 23))], :]
        return train_data, test_data

    @staticmethod
    def load_ar_exp2():
        """
        Training sample of scarf occlusion.
        :return:
        """
        data = Dataloader.load_ar()
        train_data = data[:, [*range(7), 10], :]
        test_data = data[:, [*range(11, 20), *range(23, 26)], :]
        return train_data, test_data

    @staticmethod
    def load_ar_exp3():
        """
        Training sample of sunglass and scarf occlusion.
        :return:
        """
        data = Dataloader.load_ar()
        train_data = data[:, [*range(8), 10], :]
        test_data = data[:, [8, 9, *range(11, 26)], :]
        return train_data, test_data


def make_ext_20():
    cropped_size = (20, 20)

    path = r'./data/ExtYaleB_illumination/'
    filenames_5 = os.listdir(path)
    filenames_5.sort()  # sorted by name

    data5 = []
    for filename in filenames_5:
        data = loadmat(os.path.join(path, filename))['DAT']
        a, b, c = data.shape
        imgset = data.reshape((84, 96, b, c))
        imgset_croped = np.zeros((*cropped_size, b, c))
        for i in range(b):
            for j in range(c):
                imgset_croped[:, :, i, j] = cv2.resize(imgset[:, :, i, j], cropped_size)
        data_cropped = imgset_croped.reshape((cropped_size[0] * cropped_size[1], b, c))
        data5.append(data_cropped)
    mydict = {'sub1': data5[0], 'sub2': data5[1], 'sub3': data5[2], 'sub4': data5[3], 'sub5': data5[4]}
    savemat('./data/ExtYaleB_20_20.mat', mdict=mydict)


def make_ar_20():
    cropped_size = (20, 20)

    path = r'./data/AR_120_26_50_40.mat'
    data = loadmat(path)['DAT']
    a, b, c = data.shape
    imgset = data.reshape((40, 50, b, c))
    imgset_cropped = np.zeros((*cropped_size, b, c))
    for i in range(b):
        for j in range(c):
            imgset_cropped[:, :, i, j] = cv2.resize(imgset[:, :, i, j], cropped_size)
    data_cropped = imgset_cropped.reshape((cropped_size[0] * cropped_size[1], b, c))
    mydict = {'DAT': data_cropped}
    savemat('./data/AR_120_26_20_20.mat', mdict=mydict)


def make_extyaleb_dataset():
    dataset = Dataloader.load_ExtYaleB_cropped()
    data = np.concatenate((dataset[0], dataset[1], dataset[2], dataset[3], dataset[4]), axis=1)
    train_data = data[:, [*range(8), 19, 31, 45], :]
    dataset = [data[:, 8:19, :], data[:, 20:31, :], data[:, 32:45, :], data[:, 46:, :]]
    pass


if __name__ == '__main__':
    make_ar_20()
