import os

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class Dataset:
    @staticmethod
    def formulate_data(mode, *args, shuffle_=False):
        """
        :param shuffle_:
        :param mode: '1' or '2'
            '1' represents standard mode, while '2' represents integration mode.
        :param args: data, shape like (p, a, b)
        """
        if str(mode) == '2':
            return args
        else:
            res = []
            for data_ in args:
                xs, ys = Dataset.convert_to_standard_dataset(data_, shuffle_=shuffle_)
                res += [xs, ys]
            return res

    @staticmethod
    def convert_to_standard_dataset(data_, shuffle_=False):
        """

        :param shuffle_:
        :param data_: shape like (p, a, b)
        :return:
            xs: shape like (n, p)
            ys: shape like (n, )
        """
        _, a_, b_ = data_.shape
        xs = np.vstack([data_[:, i, j] for i in range(a_) for j in range(b_)])
        ys = np.tile(np.arange(b_), a_)
        if shuffle_:
            xs, ys = shuffle(xs, ys)
        return xs, ys


class AR(Dataset):
    @staticmethod
    def _load_ar():
        path = r'data/AR/AR_120_26_20_20.mat'
        data = loadmat(path)['DAT']
        # return data
        return data / np.max(data)

    @staticmethod
    def exp1(mode='2', shuffle_=False):
        """
        Training sample of sunglass occlusion.
        :param shuffle_:
        :type mode: str, '1' or '2'
        :return:
            mode is '1': (train_xs, train_ys, test_xs, test_ys). train_xs has shape like (n_samples, n_features),
                train_ys has shape like (n_samples, )
            mode is '2': (train_data, test_data). train_data has shape like
                (n_features, n_imgs_per_class, n_classes)
        """
        data_ = AR._load_ar()
        train_data = data_[:, :8, :]
        test_data = data_[:, [8, 9, *(range(13, 23))], :]
        return AR.formulate_data(mode, train_data, test_data, shuffle_=shuffle_)

    @staticmethod
    def exp2(mode='2', shuffle_=False):
        """
        Training sample of scarf occlusion.
        """
        data_ = AR._load_ar()
        train_data = data_[:, [*range(7), 10], :]
        test_data = data_[:, [*range(11, 20), *range(23, 26)], :]
        return AR.formulate_data(mode, train_data, test_data, shuffle_=shuffle_)

    @staticmethod
    def exp3(mode='2', shuffle_=False):
        """
        Training sample of sunglass and scarf occlusion.
        """
        data_ = AR._load_ar()
        train_data = data_[:, [*range(8), 10], :]
        test_data = data_[:, [8, 9, *range(11, 26)], :]
        return AR.formulate_data(mode, train_data, test_data, shuffle_=shuffle_)


class ExtYaleB(Dataset):
    @staticmethod
    def _load_ExtYaleB(mode='1', shuffle_=False):
        """

        :param mode: '1' or '2'
            '1' represents standard mode, while '2' represents integration mode.
        :return:
            mode '1':
                [train_x, train_y, test1_x, test1_y, ..., test4_x, test4_y]
                shape like: (n_samples, n_features)
            mode '2':
                [train_data, test1_data, ..., test4_data]
                shape like: (n_features, n_imgs_per_class, n_classes)
        """

        path = r'./data/ExtYaleB_illumination/'

        file_names = os.listdir(path)
        file_names.sort()  # sorted by name

        dataset5 = []  # list
        for file_name in file_names:
            data_ = loadmat(os.path.join(path, file_name))['DAT']
            dataset5.append(data_)

        return ExtYaleB.formulate_data(mode, dataset5, shuffle_=shuffle_)

    @staticmethod
    def _load_ExtYaleB_cropped():
        """

        :return:
                [train_data, test1_data, ..., test4_data]
                shape like: (n_features, n_imgs_per_class, n_classes)
        """

        path = r'data/ExtYaleB_illumination/ExtYaleB_20_20.mat'
        dataset5_dict = loadmat(path)
        data5 = [dataset5_dict[f'sub{i}'] for i in range(1, 6)]
        return data5

    @staticmethod
    def load_ExtYaleB_20(mode=2, shuffle_=False):
        """
        subset 1全体和subset 2、3、4、5个类别第一张图片做训练集，剩余图片根据subset分为4个测试集
        :return:
            train_data, *four_test_data = load_ExtYaleB_20(mode=2)
            X_train, Y_train, *four_test_data = load_ExtYaleB_20(mode=1)
        """
        dataset5 = ExtYaleB._load_ExtYaleB_cropped()
        data_ = np.concatenate((dataset5[0], dataset5[1], dataset5[2], dataset5[3], dataset5[4]), axis=1)
        #  7  12  12  14  19
        # 0  7  19  31  45  64
        train_data = data_[:, [*range(8), 19, 31, 45], :]
        four_test_data = [data_[:, 8:19, :], data_[:, 20:31, :], data_[:, 32:45, :], data_[:, 46:, :]]
        five_data = [train_data] + four_test_data
        return ExtYaleB.formulate_data(mode, *five_data, shuffle_=shuffle_)

    @staticmethod
    def load_ExtYaleB_cropped_sub123(test_size=0.2, load_ratio=0.5):
        """
        前3份数据集混合在一起，测试集上加高斯或椒盐噪声
        :param load_ratio: float
            载入数据的比例
        :param test_size: float
            测试集比例
        :return: train_x, test_x, train_y, test_y
        """
        dataset5 = ExtYaleB._load_ExtYaleB_cropped()
        data_ = np.concatenate((dataset5[0], dataset5[1], dataset5[2]), axis=1)
        data_x, data_y = ExtYaleB.convert_to_standard_dataset(data_)

        # 数据载入比例，用于小样本超参数选择
        data_x, _, data_y, _ = train_test_split(data_x, data_y, test_size=load_ratio)
        return train_test_split(data_x, data_y, test_size=test_size)


class NUST(Dataset):
    @staticmethod
    def _load_nust():
        path = r'data/NUST/Data_indoor.mat'
        data_dict = loadmat(path)

        name6 = []
        data6 = []
        for k, v in data_dict.items():
            data_ = data_dict[k]
            h, w, a, b = data_.shape
            data_ = data_.transpose(2, 3, 0, 1).reshape(a, b, h * w).transpose(2, 0, 1)
            name6.append(k)
            data6.append(data_)
        return name6, data6

    @staticmethod
    def load_nust(mode=2, shuffle_=False):
        """

        :param shuffle_:
        :param mode: '1' or '2'
        :return:
            mode '1': [X_train, Y_train, X1_test, Y1_test, ...], five_test_name
                X_train: (n, p); Y_train: (n, 1)
            mode '2': [train_data, test1_data, ..., test5_data], five_test_name
                train_data: (p, a, b)
        """
        name6, data6 = NUST._load_nust()
        index = name6.index('Train_DAT')
        train_data = data6[index]
        five_test_data = data6.pop(index)
        five_test_name = name6.pop(index)
        return NUST.formulate_data(mode, train_data + five_test_data, shuffle_=shuffle_), five_test_name


class CUFS(Dataset):
    @staticmethod
    def _load_cufs():
        path = './data/CUFS/CUFS_cropped.mat'
        data = loadmat(path)
        photos, sketches = data['photo'], data['sketch']
        n, h, w = photos.shape
        photos = photos.reshape(n, h * w)
        sketches = sketches.reshape(n, h * w)
        # return photos / np.max(photos), sketches / np.max(sketches)  # (88, 1600)
        return photos, sketches  # (88, 1600)

    @staticmethod
    def exp1():
        """
        图片+素描训练，图片测试
        :return:
        """
        photos, sketches = CUFS._load_cufs()
        n, _ = photos.shape
        ys = np.arange(n)
        return sketches, ys, photos, ys.copy()

    @staticmethod
    def exp2():
        """
        图片+素描训练，素描测试
        :return:
        """
        pass


class Palm(Dataset):
    @staticmethod
    def _load_Palm():
        """

        :return:
            train_data: shape like (64*64, 4, 100)
            six_test_data: list
                [no occlusion, 0.1 occlusion, 0.3 occlusion, ..., 0.9 occlusion]
        """
        path = './data/Palm/Palm.mat'
        mat = loadmat(path)
        train_data = mat['Train_DAT']
        six_test_data = [mat['Test_00']]
        for i in range(1, 10, 2):
            six_test_data.append(mat[f'Test_0{i}'])

        # normalizaion
        train_data = train_data / np.max(train_data)
        for i in range(len(six_test_data)):
            six_test_data[i] = six_test_data[i] / np.max(six_test_data[i])

        return train_data, six_test_data

    @staticmethod
    def exp1(mode='2'):
        train_data, six_test_data = Palm._load_Palm()
        return Palm.formulate_data(mode, train_data, *six_test_data)


if __name__ == '__main__':
    pass
