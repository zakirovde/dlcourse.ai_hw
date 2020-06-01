import numpy as np
import math


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
    new_exp = np.vectorize(lambda x: math.exp(x))
    if len(predictions.shape)==1:
        pred = prediction-np.max(prediction)
        exp_prob = new_exp(pred)
        probs = np.array(list(map(lambda x: x/np.sum(exp_prob),exp_prob)))
    else:
        pred = list(map(lambda x: x-np.max(x), prediction))
        exp_prob = new_exp(pred)
        probs = np.array(list(map(lambda x: x/np.sum(x),exp_prob)))
    
    return probs

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
    new_loss = np.vectorize(lambda x: -math.log(x))
    if len(probs.shape)==1:
        probs_target = probs[target_index]
        size_target = 1
    else:
        batch_size = np.arange(target_index.shape[0])
        probs_target = probs[batch_size,target_index.flatten()]
        size_target = target_index.shape[0]
    loss = np.sum(new_loss(probs_target))/size_target
    return loss


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
    prediction = predictions.copy()
    probs = softmax(prediction) 
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs
    if len(predictions.shape)==1:
        dprediction[target_index] -= 1
    else:
        batch_size = np.arange(target_index.shape[0])
        dprediction[batch_size,target_index.flatten()] -= 1
        dprediction = dprediction/target_index.shape[0]

    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient
    Arguments:
      W, np array - weights
      reg_strength - float value
    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    loss = reg_strength*np.sum(np.dot(np.transpose(W),W))
    batch_size = np.arange(W.shape[1])
    grad = np.array((list(map(lambda x: np.sum(W,axis=1), batch_size))))
    grad = 2*reg_strength*np.transpose(grad)

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W
    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes
    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss
    '''
    predictions = np.dot(X, W)

    prediction = predictions.copy()
    probs = softmax(prediction) 
    loss = cross_entropy_loss(probs, target_index)
    dW = np.dot(np.transpose(X),probs)
    
    p = np.zeros_like(probs)
    batch_size = np.arange(target_index.shape[0])
    p[batch_size,target_index.flatten()] = 1
    dW -= np.transpose(np.dot(np.transpose(p),X))
    dW = dW/target_index.shape[0]
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = np.arange(epochs).astype(np.float)
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            
            for i in range(sections.shape[0]): 
                batch = X[batches_indices[i],:]
                target_index = y[batches_indices[i]]
                loss_W, dW = linear_softmax(batch, self.W, target_index)
                loss_l, dl = l2_regularization(self.W, reg)
                self.W = self.W - learning_rate*(dW+dl)
                loss = loss_W+loss_l
                
            loss_history[epoch] = loss
            #print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)
        Returns:
          y_pred, np.array of int (test_samples)
        '''
        #y_pred = np.zeros(X.shape[0], dtype=np.int)
        prediction = np.dot(X, self.W)
        probs = softmax(prediction)
        y_pred = np.array(list(map(lambda x: x.argsort()[-1], probs)))
        
        return y_pred