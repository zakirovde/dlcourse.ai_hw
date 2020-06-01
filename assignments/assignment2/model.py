import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.layer_1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.layer_2 = ReLULayer()
        self.layer_3 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        params = self.params()
        
        for p in params:
            param = params[p]
            param.grad = np.zeros_like(param.grad)
            
        step_1f = self.layer_1.forward(X)
        step_2f = self.layer_2.forward(step_1f)
        step_3f = self.layer_3.forward(step_2f)
        
        loss, dpred = softmax_with_cross_entropy(step_3f, y)
        
        step_3b = self.layer_3.backward(dpred)
        step_2b = self.layer_2.backward(step_3b)
        step_1b = self.layer_1.backward(step_2b)
        
        for p in params:
            param = params[p]
            loss_l2, grad_l2 = l2_regularization(param.value, self.reg)
            param.grad += grad_l2
            loss += loss_l2
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        step_1f = self.layer_1.forward(X)
        step_2f = self.layer_2.forward(step_1f)
        step_3f = self.layer_3.forward(step_2f)
        probs = softmax(step_3f)
        pred = np.array(list(map(lambda x: x.argsort()[-1], probs)))
        return pred

    def params(self):
        result = {}
        result['W1'] = self.layer_1.W
        result['W2'] = self.layer_3.W
        result['B1'] = self.layer_1.B
        result['B2'] = self.layer_3.B
        return result
