#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-03 13:48:21
# @Author  : Paul (pz.suen@gmail.com)
# @Link    : ${link}
# @Version : $Id$

import random
import numpy as np
import matplotlib.pyplot as plt
import h5py


def load_dataset():
    train_dataset = h5py.File('datasets/train_cifar10.h5', "r")
    train_set_x_orig = np.array(train_dataset["X_train"][
                                :])  # your train set features
    train_set_y_orig = np.array(train_dataset["y_train"][
                                :])  # your train set labels

    test_dataset = h5py.File('datasets/test_cifar10.h5', "r")
    test_set_x_orig = np.array(test_dataset["X_test"][
                               :])  # your test set features
    test_set_y_orig = np.array(test_dataset["y_test"][
                               :])  # your test set labels

    classes = np.array(train_dataset["classes_list"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def compute_distances(X_train, X_test):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X_test.shape[0]
    num_train = X_train.shape[0]
    dists = np.zeros((num_test, num_train))

    # (a-b)^2=a^2-2ab+b^2
    # 获取相乘后对角线的值，即为每一行aa(1,num_test)和bb(1,num_train)的和
    aa = np.diag(np.dot(X_test, X_test.T)).reshape(1, num_test)
    bb = np.diag(np.dot(X_train, X_train.T)).reshape(1, num_train)
    # 计算ab的值 ,shape 为 (num_test,num_train)
    ab = np.dot(X_test, X_train.T)
    dists = np.sqrt(aa.T - 2 * ab + bb)

    return dists


def predict_labels(y_train, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    # print(num_test)
    for i in range(num_test):
        # A list of length k storing the labels of the k nearest neighbors to
        # the ith test point.
        closest_y = []
        count = np.zeros(11)  # 用于存储标签统计列表，没一个test都要更新

        tags = np.argsort(dists[i, :])[:k]

        closest_y = y_train[tags]

        if k == 1:
            y_pred[i] = closest_y[0]
        else:
            for ki in range(k):
                count[closest_y[ki]] += 1               # 每个分类出现一次，其count值+1
            y_pred[i] = np.argsort(count)[-1]         # 选择最大的下标，即为其分类

    return y_pred

if __name__ == "__main__":

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    # print("X_train_orig shape:", X_train_orig.shape)
    # print("Y_train_orig shape:", Y_train_orig.shape)
    # print("X_test_orig shape:", X_test_orig.shape)
    # print("Y_test_orig shape:", Y_test_orig.shape)

    # Flatten the training and test images
    X_train_flatten = X_train_orig.reshape(-1, X_train_orig.shape[0]).T
    X_test_flatten = X_test_orig.reshape(-1, X_test_orig.shape[0]).T
    # 正则化
    X_train_flatten = X_train_flatten / 255.
    X_test_flatten = X_test_flatten / 255.

    # 下采样，只取1/10的训练样本，1/20的测试样本（50000*10000的矩阵太大）
    num_training = 5000
    mask = list(range(num_training))
    X_train_flatten = X_train_flatten[mask]
    Y_train_orig = Y_train_orig[0][mask].T

    num_test = 500
    mask = list(range(num_test))
    X_test_flatten = X_test_flatten[mask]
    Y_test_orig = Y_test_orig[0][mask].T

    # (5000, 3072) (500, 3072)
    print(X_train_flatten.shape, X_test_flatten.shape)
    print(Y_train_orig.shape, Y_test_orig.shape)

    # 计算距离，即训练(500,5000)
    dists = compute_distances(X_train_flatten, X_test_flatten)

    # 预测
    Y_test_pred = predict_labels(Y_train_orig, dists, k=2)

    # Compute and display the accuracy
    num_correct = np.sum(Y_test_pred == Y_test_orig)
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
