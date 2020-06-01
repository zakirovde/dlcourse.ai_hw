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
