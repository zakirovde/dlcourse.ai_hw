import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    loss = reg_strength * np.sum(W*W)
    grad = 2*reg_strength*W
    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    prediction = predictions.copy()
    if len(prediction.shape)==1:
        prediction -= np.max(prediction)
        s = np.exp(prediction) / np.sum(np.exp(prediction))
    else:
        prediction -= np.reshape(np.max(prediction, axis=1), (-1,1))
        s = np.exp(prediction)/np.reshape(np.exp(prediction).sum(1), (-1,1))
    return s


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    if len(probs.shape)==1:
        l = -np.log(probs[target_index])
    else:
        l = -np.log(probs)[np.arange(probs.shape[0]), target_index.flatten()]
        l = l.sum() / len(l)
    return l


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    prediction = preds.copy()
    probs = softmax(prediction)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs
    if len(probs.shape)==1:
        dprediction[target_index] -= 1
    else:
        dprediction[np.arange(probs.shape[0]), target_index.flatten()] -= 1
        dprediction /= probs.shape[0]
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        X_new = X.copy()
        X_new[X_new<0] = 0
        return X_new

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        d_result = d_out
        d_result[self.X<0] = 0
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        f = np.dot(X, self.W.value) + self.B.value
        return f

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        d_input = np.dot(d_out, self.W.value.T)
        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += np.sum(d_out, axis=0)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X
        if self.padding > 0:
            self.Xp = np.zeros((batch_size, height + 2 * self.padding,
                                width + 2 * self.padding, channels))
            self.Xp[:, self.padding : self.padding + height,
                    self.padding : self.padding + width, :] = self.X
        else:
            self.Xp = self.X
        stride = 1
        out_height = int((height - self.filter_size + 2 * self.padding) / stride + 1)
        out_width = int((width - self.filter_size + 2 * self.padding) / stride + 1)
        
        result = np.zeros((batch_size, out_height, out_width, self.out_channels))
        W_dash = self.W.value.reshape(self.filter_size * self.filter_size * self.in_channels,
                                      self.out_channels)
        B_times = np.zeros((batch_size, self.out_channels))
        B_dash = np.array(list(map(lambda x: self.B.value, B_times)))
        for y in range(out_height):
            for x in range(out_width):
                X_dash = self.Xp[:, y : y + self.filter_size,
                                 x : x + self.filter_size, :]
                X_dash = X_dash.reshape(batch_size
                                        , self.filter_size * self.filter_size * self.in_channels)
                result[:, y, x, :] = np.dot(X_dash, W_dash) + B_dash
        return result


    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        d_in_p = np.zeros(self.Xp.shape)
        W_dash = self.W.value.reshape(self.filter_size * self.filter_size * channels, out_channels)
        for y in range(out_height):
            for x in range(out_width):
                d_out_dash = d_out[:, y, x, :].reshape(batch_size, out_channels)
                X_dash = self.Xp[:, y : y + self.filter_size, x : x + self.filter_size, :]
                d_out_add = np.dot(d_out_dash, W_dash.T).reshape(X_dash.shape)
                d_in_p[:, y : y + self.filter_size, x : x + self.filter_size, :] += d_out_add
                X_dash = X_dash.reshape(batch_size, self.filter_size * self.filter_size * channels)
                self.W.grad += np.dot(X_dash.T, d_out_dash).reshape(self.W.value.shape)
                self.B.grad += np.sum(d_out_dash, axis=0)
        if self.padding > 0:
            d_input = d_in_p[:, self.padding : self.padding + height
                             , self.padding : self.padding + width, :]
        else:
            d_input = d_in_p
        return d_input
            

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X
        out_height = int((height - self.pool_size) / self.stride + 1)
        out_width = int((width - self.pool_size) / self.stride + 1)
        result = np.zeros((batch_size, out_height, out_width, channels))
        self.d_input = np.zeros_like(self.X)
        for y in range(out_height):
            for x in range(out_width):
                step_x = x * self.stride
                step_y = y * self.stride
                X_max = X[:, step_y : step_y + self.pool_size, step_x : step_x + self.pool_size, :]
                X_max = X_max.reshape(batch_size * channels, self.pool_size ** 2)
                X_max_val = np.max(X_max, axis=1)
                result[:, y, x, :] = X_max_val.reshape(batch_size, channels)
                X_max_arg = np.argmax(X_max, axis=1)
                d_input = np.zeros_like(X_max)
                for i in range(batch_size * channels):
                    d_input[i, X_max[i,:]==X_max_val[i]] += 1
                d_input = np.array(list(map(lambda x: x/np.sum(x), d_input)))
                self.d_input[:, step_y : step_y + self.pool_size, step_x : step_x + self.pool_size, :] \
                    = d_input.reshape(batch_size, self.pool_size, self.pool_size, channels)
        return result

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        d_input = self.d_input
        for y in range(out_height):
            for x in range(out_width):
                step_x = x * self.stride
                step_y = y * self.stride
                d_out_tmp = d_out[:,x,y,:].reshape(batch_size*channels)
                d_out_add = np.array(list(map(lambda x: np.ones(self.pool_size * self.pool_size) * d_out_tmp[x]
                                              , np.arange(batch_size * channels))))
                d_out_add = d_out_add.reshape(batch_size, self.pool_size, self.pool_size, channels)
                d_input[:,step_x : step_x + self.pool_size, step_y : step_y + self.pool_size, :] *= d_out_add
        return d_input

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.channels = channels
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        return d_out.reshape(self.batch_size, self.height, self.width, self.channels)

    def params(self):
        # No params!
        return {}
