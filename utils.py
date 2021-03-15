import numpy as np


def addsalt_pepper(img, SNR):
    img_ = img.copy()
    h, w = img_.shape
    mask = np.random.choice((0, 1, 2), size=(h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    # img_[mask == 1] = np.max(img)  # 盐噪声
    # img_[mask == 2] = np.min(img)  # 椒噪声
    img_[mask == 1] = 1  # 盐噪声
    img_[mask == 2] = 0  # 椒噪声
    return img_


def batch_addsalt_pepper(dataset, SNR, shape=(20, 20)):
    dataset_ = dataset.copy()
    n, p = dataset.shape
    if p != shape[0] * shape[1]:
        pass
    for i in range(n):
        img = dataset[i].reshape(shape)
        img_ = addsalt_pepper(img, SNR)
        dataset_[i] = img_.flatten()
    return dataset_
