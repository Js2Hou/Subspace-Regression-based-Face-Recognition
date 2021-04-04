# -*- coding: utf-8 -*-
"""
@Time ： 2021/3/26 15:35
@Auth ： JsHou
@Email : 137073284@qq.com
@File ：multiprocess.py

"""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool


def func(x):
    return x ** 2


ls = []


def cb(data):
    ls.append(data)


if __name__ == '__main__':
    x = list(range(4))
    with ProcessPoolExecutor() as pool:
        res = list(pool.map(func, x))
    print(res)

    y = [i for i in range(4)]
    pool = Pool()
    for e in y:
        pool.apply_async(func=func, args=(e,), callback=cb)
    pool.close()
    pool.join()
    print(ls)
