import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization, softmax
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        self.layer_1 = ConvolutionalLayer(input_shape[2], conv1_channels, 3, 1)
        self.layer_2 = ReLULayer()
        self.layer_3 = MaxPoolingLayer(4, 4)
        self.layer_4 = ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1)
        self.layer_5 = ReLULayer()
        self.layer_6 = MaxPoolingLayer(4, 4)
        self.layer_7 = Flattener()
        self.layer_8 = FullyConnectedLayer((input_shape[0] * input_shape[1] * conv2_channels) // (16 ** 2), n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        params = self.params()
        
        for param_key in params:
            param = params[param_key]
            param.grad = np.zeros_like(param.grad)
            
        step_1f = self.layer_1.forward(X)
        step_2f = self.layer_2.forward(step_1f)
        step_3f = self.layer_3.forward(step_2f)
        step_4f = self.layer_4.forward(step_3f)
        step_5f = self.layer_5.forward(step_4f)
        step_6f = self.layer_6.forward(step_5f)
        step_7f = self.layer_7.forward(step_6f)
        step_8f = self.layer_8.forward(step_7f)
        
        loss, dpred = softmax_with_cross_entropy(step_8f, y)
        
        step_8b = self.layer_8.backward(dpred)
        step_7b = self.layer_7.backward(step_8b)
        step_6b = self.layer_6.backward(step_7b)
        step_5b = self.layer_5.backward(step_6b)
        step_4b = self.layer_4.backward(step_5b)
        step_3b = self.layer_3.backward(step_4b)
        step_2b = self.layer_2.backward(step_3b)
        step_1b = self.layer_1.backward(step_2b)
        
        return loss
    
    def predict(self, X):
        step_1f = self.layer_1.forward(X)
        step_2f = self.layer_2.forward(step_1f)
        step_3f = self.layer_3.forward(step_2f)
        step_4f = self.layer_4.forward(step_3f)
        step_5f = self.layer_5.forward(step_4f)
        step_6f = self.layer_6.forward(step_5f)
        step_7f = self.layer_7.forward(step_6f)
        step_8f = self.layer_8.forward(step_7f)
        probs = softmax(step_8f)
        pred = np.array(list(map(lambda x: x.argsort()[-1], probs)))
        return pred

    def params(self):
        result = {}
        result['W1'] = self.layer_1.W
        result['W2'] = self.layer_4.W
        result['W3'] = self.layer_8.W
        result['B1'] = self.layer_1.B
        result['B2'] = self.layer_4.B
        result['B3'] = self.layer_8.B
        return result
