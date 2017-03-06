#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-3-4 上午10:54
# @Author  : ywl
# @File    : data_gen.py
import os

os.chdir('/home/pig/PycharmProjects/Graduate/bishe')
import arff
import numpy as np


def gen_label_one_hot(label_index, num_labels):
    result = []
    y = []

    if len(label_index) != 0:
        for i in label_index:
            temp = np.zeros([num_labels])
            temp[i] = 1
            result.append(temp)
            y.append(np.array(1))
    for rest in range(num_labels - len(label_index)):
        result.append(np.zeros([num_labels]))
        y.append(np.array(0))

    return np.array(result), np.array(y)


def gen_X_y(dataset_name):
    data_origin = arff.load(open('dataset/' + dataset_name + '.arff', 'rb'))
    data = [[int(i) for i in line] for line in data_origin['data']]
    num_labels = 208
    num_data = len(data)
    num_attributes = data_origin['attributes'].__len__() - num_labels
    # num_attributes为2150+208,最终为2358个

    data_label = np.array([line[-208:] for line in data])
    data_shuju = np.array([line[:-208] for line in data])

    data_label_statistics = []

    label_sum = map(sum, zip(*data_label))
    # 存放每个类标出现的次数
    dict_label_sum = {}
    for i in range(num_labels):
        dict_label_sum[i] = label_sum[i]

    label_quantity_sorted = [i[0] for i in sorted(dict_label_sum.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)]

    for i in range(num_data):
        data_label[i] = data_label[i][np.array(label_quantity_sorted)]

    X = []
    y = []
    for i in range(num_data):

        label_index = np.where(data_label[i] == 1)[0]
        label_mat, temp_y = gen_label_one_hot(label_index, num_labels)

        y.append(temp_y)

        temp_X = []
        for j in range(num_labels):
            # temp_X = temp_X + list(data_shuju[i]) + list(label_mat[j])
            temp_X = temp_X + list(data_shuju[i])
        X.append(temp_X)
    X = np.array(X)
    y = np.array(y)
    return X, y


class dataset():
    def __init__(self, X, y):
        self.labels = y
        self.data = X
        self.num_samples = len(X)
        self.current = 0

    def next_batch(self, batch):
        if self.current + batch > self.num_samples:
            self.current = (self.current + batch) % self.num_samples

        X_ = self.data[self.current:self.current + batch]
        y_ = self.labels[self.current:self.current + batch]
        self.current += batch
        return X_, y_


# mnist = dataset(*gen_X_y('bookmarks1'))
# import cPickle as pickle
#
# f = open('book.pkl', 'w')
# pickle.dump(mnist,f)
