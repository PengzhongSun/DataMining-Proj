#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-03 14:25:26
# @Author  : Paul (pz.suen@gmail.com)

import random
import numpy as np
import h5py
from copy import deepcopy
import matplotlib.pyplot as plt

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


def normalize_X():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    Y_train_orig=Y_train_orig.T
    Y_test_orig=Y_test_orig.T
    # print("X_train_orig shape:", X_train_orig.shape)
    # print("Y_train_orig shape:", Y_train_orig.shape)
    # print("X_test_orig shape:", X_test_orig.shape)
    # print("Y_test_orig shape:", Y_test_orig.shape)
    # X_train_orig shape: (50000, 32, 32, 3)
    # Y_train_orig shape: (1, 50000)
    # X_test_orig shape: (10000, 32, 32, 3)
    # Y_test_orig shape: (1, 10000)

    # Split the data into train, val, and test sets. In addition we will
    # create a small development set as a subset of the training data;
    # we can use this for development so our code runs faster.
    num_training = 49000
    num_validation = 1000
    num_test = 1000
    num_dev = 500

    # Our validation set will be num_validation points from the original
    # training set.
    # 49000——50000
    mask = range(num_training, num_training + num_validation)
    X_val = X_train_orig[mask]
    y_val = Y_train_orig[mask]

    # Our training set will be the first num_train points from the original
    # training set.
    # 0——49000
    mask = range(num_training)
    X_train = X_train_orig[mask]
    y_train = Y_train_orig[mask]

    # We will also make a development set, which is a small subset of
    # the training set.
    # 从训练集从随机选500个
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train_orig[mask]
    y_dev = Y_train_orig[mask]

    # We use the first num_test points of the original test set as our
    # test set.
    # 测试集只用了原测试集的1/10
    mask = range(num_test)
    X_test = X_test_orig[mask]
    y_test = Y_test_orig[mask]

    # print('Train data shape: ', X_train_orig.shape)
    # print('Train labels shape: ', Y_train_orig.shape)
    # print('Test data shape: ', X_test_orig.shape)
    # print('Test labels shape: ', Y_test_orig.shape)
  

    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

    # As a sanity check, print out the shapes of the data
    # print('Training data shape: ', X_train.shape)
    # print('Validation data shape: ', X_val.shape)
    # print('Test data shape: ', X_test.shape)
    # print('dev data shape: ', X_dev.shape)

    # 预处理: subtract the mean image
    # first: compute the image mean based on the training data
    mean_image = np.mean(X_train, axis=0).T.reshape(1,-1).astype(np.uint8)
    # print(X_train.dtype)
    # second: subtract the mean image from train and test data
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image
    # third: append the bias dimension of ones (i.e. bias trick) so that our SVM
    # only has to worry about optimizing a single weight matrix W.
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    return X_train,y_train, X_val,y_val, X_test,y_test, X_dev,y_dev


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = np.dot(X, W)
    y = y.reshape(X.shape[0], 1)
    # real_class=np.array(np.arange(X.shape[0]),y)
    # 计算每个样本对英语每个分类的得分
    mask = (scores - scores[np.arange(num_train),
                            y[:, 0]].reshape(-1, 1) + 1) > 0
    # margins = scores - scores[np.arange(num_train),y] + 1
    margins = scores - scores[np.arange(num_train), y[:, 0]].reshape(-1, 1) + 1
    # 将真实类别和得分小于零的项置零
    margins[np.arange(num_train), y[:, 0]] = 0
    margins *= mask
    # 计算损失
    loss = np.sum(margins)
    loss /= num_train
    loss += reg * np.sum(W * W)
    # print(loss)

    # dw=x.T*dl/ds
    ds = np.ones_like(scores)  # 初始化ds
    ds = ds * mask             # 有效的score梯度为1，无效的为0
    # 真实类别要减去j次X（j为score>0的元素个数）、
    # 后面减一是因为mask每行多一个真实类别的1
    ds[np.arange(num_train), y[:, 0]] = -1 * (np.sum(ds, axis=1) - 1)
    dW = np.dot(X.T, ds)
    dW /= num_train
    dW += 2 * reg * W

    return loss, dW


def train(W,X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
          batch_size=200, verbose=False):
    """
    Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.
    """
    num_train, dim = X.shape   # num_train=N ,dim =D
    # assume y takes values 0...K-1 where K is number of classes
    num_classes = np.max(y) + 1
    if W is None:
        # lazily initialize W
        W = 0.001 * np.random.randn(dim, num_classes)

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in range(num_iters):
        X_batch = None
        y_batch = None

        # X_batch : (dim, batch_size)
        # y_batch : (batch_size,)
        indices = np.random.choice(X.shape[0], batch_size, replace=True)
        X_batch = X[indices, :]
        y_batch = y[indices]

        # evaluate loss and gradient
        # 调用自己的函数进行运算，返回损失和梯度
        loss, grad = svm_loss_vectorized(W,X_batch, y_batch, reg)
        loss_history.append(loss)

        # perform parameter update
        W = W - learning_rate * grad

        if verbose and it % 100 == 0:
            print('iteration %d / %d: loss %f' % (it, num_iters, loss))

    return loss_history,W


def predict(X,W):
    y_pred = np.zeros(X.shape[0])
    scores = np.dot(X, W)
    y_pred = np.argmax(scores, axis=1)

    return y_pred


if __name__ == "__main__":
    X_train,y_train, X_val,y_val, X_test,y_test, X_dev,y_dev = normalize_X()
    print(X_train.shape, X_val.shape, X_test.shape, X_dev.shape)

    W = np.random.randn(3073, 10) * 0.0001

    loss_hist,trained_W = train(W,X_train, y_train, learning_rate=1e-7,reg=7.5e4, num_iters=1500, verbose=True)

    # 画出loss 的图
    plt.plot(loss_hist)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()

    # 测试相对于training and validation set的准确率
    y_train_pred = predict(X_train,trained_W)
    print('training accuracy: %f' % (np.mean(y_train == y_train_pred), ))
    y_val_pred = predict(X_val,trained_W)
    print('validation accuracy: %f' % (np.mean(y_val == y_val_pred), ))
