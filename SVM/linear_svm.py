import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]      # 类别个数
    num_train = X.shape[0]        # 训练样本个数
    loss = 0.0
    for i in range(num_train):  # 遍历每个样本
        # shape=(1,c),其中c为类别,计算各个类别对该样本的打分scores
        scores = np.dot(X[i], W)
        # 其真实类别对该样本的打分,y[i]为其真实类别，即为下标
        correct_class_score = scores[y[i]]
        for j in range(num_classes):  # 计算该样本的损失
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, y[i]] += -X[i].T
                dW[:, j] += X[i].T

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    dW += 2 * reg * W
    ##########################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    ##########################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    ##########################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    ##########################################################################
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

    ##########################################################################
    #                             END OF YOUR CODE                              #
    ##########################################################################

    ##########################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    ##########################################################################
    # pass
    ##########################################################################
    #                             END OF YOUR CODE                              #
    ##########################################################################

    return loss, dW


if __name__ == "__main__":
    np.random.seed(1)
    W = np.arange(73 * 10).reshape(73, 10)
    X = np.arange(49 * 73).reshape(49, 73)
    y = np.random.randint(0, 9, (49, 1))
#   print(y)
    reg = 1
    loss, dW = svm_loss_naive(W, X, y, reg)
    print(loss, dW)
