from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.
    相当于两层网络：affine（relu），affine（softmax）

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        #######################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        #######################################################################
        # 这句话至关重要Gaussian centered at 0.0，standard deviation equal to weight_scale
        # W1 = np.random.randn(input_dim, hidden_dim) * weight_scale
        W1 = np.random.normal(0, weight_scale, size=(input_dim, hidden_dim))
        b1 = np.zeros((1, hidden_dim))
        W2 = np.random.normal(0, weight_scale, size=(hidden_dim, num_classes))
        b2 = np.zeros((1, num_classes))

        # self.params{"W1"}=W1
        self.params["W1"] = W1
        self.params["b1"] = b1
        self.params["W2"] = W2
        self.params["b2"] = b2
        #######################################################################
        #                             END OF YOUR CODE                             #
        #######################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        # 如果y==None，是测试
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        # 如果y！=None，是训练
        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        #######################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        #######################################################################
        W1 = self.params["W1"]
        b1 = self.params["b1"]
        W2 = self.params["W2"]
        b2 = self.params["b2"]
        # print(W1,b1,W2,b2)

        # 第一层
        # print("X:" + str(X.shape))
        # print("W1:" + str(W1.shape))
        # print("b1" + str(b1.shape))
        A1, cache1 = affine_relu_forward(X, W1, b1)
        # print("cache1:" + str(cache1.shape))
        # print("A1:" + str(A1.shape))
        fc_cache, relu_cache = cache1  # fc_cache = (x, w, b); relu_cache=Z1
        # W1 = fc_cache[1]
        # b1 = fc_cache[2]

        # 第二层
        scores, cache2 = affine_forward(A1, W2, b2)      # cache = (x, w, b)
        # W2 = cache2[1]
        # b2 = cache2[2]

        #######################################################################
        #                             END OF YOUR CODE                             #
        #######################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        #######################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        #######################################################################
        N = X.shape[0]
        # 计算loss
        # loss,dx=softmax_loss(scores,y)
        scores -= np.max(scores, axis=1, keepdims=True)      # 数值稳定性
        scores = np.exp(scores)
        scores = scores / np.sum(scores, axis=1).reshape(scores.shape[0], -1)
        real_y_score = scores[np.arange(N), y]
        L_i = -np.log(real_y_score)
        loss = np.sum(L_i) / X.shape[0]
        loss += 0.5 * self.reg * (np.sum(W2 * W2) + np.sum(W1 * W1))

        # # 计算第二层
        scores_copy = np.copy(scores)
        scores_copy[np.arange(N), y] -= 1
        dW2 = np.dot(A1.reshape(A1.shape[0], -1).T, scores_copy) / N
        # db2 = np.sum(scores_copy, axis=0) / X.shape[0]
        db2 = np.sum(scores_copy, axis=0) / N

        # # 正则化
        dW2 += self.reg * W2

        # # 为了反向传播是上一层（第一层）使用
        dA1 = np.dot(scores_copy, W2.T).reshape(A1.shape)

        dZ1 = dA1 * (relu_cache > 0)
        dW1 = np.dot(X.reshape(N, -1).T, dZ1) / N
        db1 = np.sum(dZ1, axis=0) / N
        dW1 += self.reg * W1

        grads["W1"] = dW1
        grads["b1"] = db1
        grads["W2"] = dW2
        grads["b2"] = db2
        #######################################################################
        #                             END OF YOUR CODE                             #
        #######################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        # print("Start initialize the FullyConnectedNet.....")
        self.normalization = normalization
        self.use_dropout = dropout != 1  # (dropout默认为1, (dropout != 1)=False, 因此不实用dropout)
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)   # 层数为隐藏层数+1（输出层）
        self.dtype = dtype
        self.params = {}

        #######################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        #######################################################################
        # {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
        # 加设hidden_layers=5,则共有5个隐藏层（第一个隐藏层接收输入，最后一层输出)
        # layer_i:0,1,2,3,4
        # params中层的值应该为layer_i+1:1,2,3,4,5；第1和第5层区别对待

        for layer_i in range(self.num_layers-1):    # 前4层的处理
            if layer_i == 0:            # 第一个hidden layer是第0个
                self.params["W" + str(layer_i + 1)] = np.random.normal(0,weight_scale, size=(input_dim, hidden_dims[layer_i]))
                self.params["b" + str(layer_i + 1)] = np.zeros((1, hidden_dims[layer_i]))
                if self.normalization is not None:
                    self.params[
                        "gamma" + str(layer_i + 1)] = np.ones((1, hidden_dims[layer_i]))
                    self.params[
                        "beta" + str(layer_i + 1)] = np.zeros((1, hidden_dims[layer_i]))
            else:
                self.params["W" + str(layer_i + 1)] = np.random.normal(
                    0, weight_scale, size=(hidden_dims[layer_i - 1], hidden_dims[layer_i]))
                self.params["b" + str(layer_i + 1)
                            ] = np.zeros((1, hidden_dims[layer_i]))
                if self.normalization is not None:
                    self.params[
                        "gamma" + str(layer_i + 1)] = np.ones((1, hidden_dims[layer_i]))
                    self.params[
                        "beta" + str(layer_i + 1)] = np.zeros((1, hidden_dims[layer_i]))
            # print("The initialized W of layer ",layer_i+1," is ",self.params['W'+str(layer_i+1)].shape)
            # print("The initialized b of layer ",layer_i+1," is ",self.params['b'+str(layer_i+1)].shape)

        self.params["W" + str(self.num_layers)] = np.random.normal(0, weight_scale, size=(hidden_dims[-1], num_classes))
        self.params["b" + str(self.num_layers)] = np.zeros((1, num_classes))
        # print("The initialized W of layer ",self.num_layers," is ",self.params['W'+str(self.num_layers)].shape)
        # print("The initialized b of layer ",self.num_layers," is ",self.params['b'+str(self.num_layers)].shape)

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:  # 这个seed是传进来的参数
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == 'batchnorm':
            self.bn_params = [{'mode': 'train'}
                              for i in range(self.num_layers - 1)]
        if self.normalization == 'layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        # 如果y==None，是测试
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        # 如果y！=None，是训练
        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'  # y=None 说明是test

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization == 'batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        #######################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        #######################################################################
        # 前向传播
        scores=X
        caches=list()   #总的cache，元素个数为层数
        for layer_i in range(self.num_layers - 1):
            # layer指当前层
            # print("layer_i=",layer_i)
            layer = str(layer_i + 1)
            cache=list()            # 每层的cache集合
            # af_cache = (x, w, b)
            # print("This is layer "+str(layer)+".........")
            # print("X=",X.shape,"W=",self.params['W' + layer].shape, 'b=',self.params["b" + layer].shape)
            scores, af_cache = affine_forward(scores, self.params['W' + layer], self.params["b" + layer])
            cache.append(af_cache)
            # 在激活之间进行归一化
            if self.normalization == 'batchnorm':
                gamma=self.params["gamma"+layer]
                beta=self.params['beta'+layer]
                scores,bn_cache = batchnorm_forward(scores,gamma,beta,self.bn_params[layer_i])
                cache.append(bn_cache)
            elif self.normalization == 'layernorm':
                gamma=self.params["gamma"+layer]
                beta=self.params['beta'+layer]
                scores,ln_cache = layernorm_forward(scores,gamma,beta,self.ln_params[layer_i])
                cache.append(ln_cache)
            # 激活
            # relu_cache = x
            scores, relu_cache = relu_forward(scores)
            cache.append(relu_cache)                
            # dropout应当在激活函数之后
            if self.use_dropout:
                scores,dp_cache=dropout_forward(scores,self.dropout_param)
                cache.append(dp_cache)
            caches.append(cache)    
        # 最后一层 ，不再需要归一化             
        last_layer = str(self.num_layers)
        # fc_cache = (x, w, b)
        scores,af_cache=affine_forward(scores,self.params["W"+last_layer],self.params["b"+last_layer])
        caches.append(af_cache)
        #######################################################################
        #                             END OF YOUR CODE                             #
        #######################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        #######################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        #######################################################################
        # 1、正则化,记得乘以0.5
        # 2、反向传播得到的dw、db存入grads
        loss,dx=softmax_loss(scores,y)
        for layer_i in range(1,self.num_layers+1):
            loss += 0.5 * self.reg * np.sum(self.params["W"+str(layer_i)]**2)

        # 对最后一层反向传播
        af_cache=caches[-1]         
        dx,dw,db=affine_backward(dx,af_cache)      # af_cache = x,w,b
        grads['W'+last_layer] = dw + self.reg * self.params['W'+last_layer]
        grads['b'+last_layer] = db

        for layer_i in range(self.num_layers-1,0,-1):   # 有num_layers-1没有0
            if self.use_dropout:
                dx = dropout_backward(dx,caches[layer_i-1][-1])

            # 有无正则化是两种情况，因为caches内容不同，分别考虑
            if self.normalization is not None:
                dx = relu_backward(dx,caches[layer_i-1][2])
                if self.normalization=="batchnorm":
                     dx, dgamma, dbeta=batchnorm_backward(dx,caches[layer_i-1][1])
                elif self.normalization == 'layernorm':
                    dx, dgamma, dbeta = layernorm_backward(dx,caches[layer_i-1][1])
                else:
                    raise ValueError("There is no such normalization!")
                grads["gamma"+str(layer_i)]=dgamma
                grads["beta"+str(layer_i)]=dbeta
            else:
                dx = relu_backward(dx,caches[layer_i-1][1])
            dx,dw,db = affine_backward(dx,caches[layer_i-1][0])
            grads['W'+str(layer_i)] = dw + self.reg * self.params['W'+str(layer_i)]
            grads['b'+str(layer_i)] = db
        #######################################################################
        #                             END OF YOUR CODE                             #
        #######################################################################

        return loss, grads
