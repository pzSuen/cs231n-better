import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ##########################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    ##########################################################################
    num_example = X.shape[0]
    num_class = W.shape[1]
    # y = y.reshape(num_example, 1)
#   S = np.array(num_X).reshape(num_X,1)
    for i in range(num_example):              # 遍历所有样本
        # score = np.dot(X[i], W)
        # score -= max(score)  # 为了数值稳定性
        # score = np.exp(score)  # 取指数
        # softmax_sum = np.sum(score)  # 得到分母
        # score /= softmax_sum  # 除以分母得到softmax
        # # 计算梯度
        # for j in range(W.shape[1]):
        #     if j != y[i]:
        #         dW[:, j] += score[j] * X[i]
        #     else:
        #         dW[:, j] -= (1 - score[j]) * X[i]

        # loss -= np.log(score[y[i]])  # 得到交叉熵
        # print(loss)

        scores_sum = 0.0                      # 存储各个类别的总和
        scores_i = np.dot(X[i, :], W)
        scores_i -= max(scores_i)                 # 为了数值的稳定性
        scores_i = np.exp(scores_i)
        scores_sum = np.sum(scores_i)             # 并计算总的score
        scores_i /= scores_sum          # 真实类别所占的比例
        loss_i = -np.log(scores_i[y[i]])        # 第i个样本的损失
        loss += loss_i
        # 计算dW
        for j in range(num_class):
            if j != y[i]:
                dW[:, j] += scores_i[j] * X[i, :]
            else:
                dW[:, j] += (scores_i[j] - 1) * X[i, :]
        # print(str(i)+":"+str(loss))

    ##########################################################################
    #                          END OF YOUR CODE                                 #
    ##########################################################################
    loss /= num_example
    dW /= num_example

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    # print(loss.shape)
    # print(dW.shape)
    # print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")

    # print(dW)

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.

    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ##########################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    ##########################################################################
    N = X.shape[0]
    scores = np.dot(X, W)
    scores = scores - np.max(scores,axis=1,keepdims=True)  # 数值稳定性
    scores = np.exp(scores)
    scores = scores / np.sum(scores, axis=1, keepdims=True)

    loss_i = scores[np.arange(N), y]
    loss_i = -np.log(loss_i).sum()
    loss = np.sum(loss_i) / N
    loss += reg * np.sum(W * W)


    ds = np.copy(scores)
    ds[np.arange(N), y] -= 1

    dW += np.dot(X.T,ds)
    dW/=N
    dW += 2 * reg * W
    # print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")

    # print(dW)

    ##########################################################################
    #                          END OF YOUR CODE                                 #
    ##########################################################################

    return loss, dW
