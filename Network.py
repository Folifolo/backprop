import numpy as np
from utils import *

class NN:
    def __init__(self, input_size = 784, hidden_size=30, output_size=10):
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size

        self._w1 = np.array([])
        self._w2 = np.array([])

        self._b1 = np.array([])
        self._b2 = np.array([])

        self._output1 = np.array([])
        self._output2 = np.array([])

        self._dE1 = np.array([])
        self._dE2 = np.array([])

        self._db2 = np.array([])
        self._db1 = np.array([])

    def init_weights(self):
        self._w1 =np.random.randn(self._hidden_size, self._input_size)/10
        self._w2 =np.random.randn(self._output_size, self._hidden_size)/10

        self._b1 = np.zeros((self._hidden_size,1))
        self._b2 = np.zeros((self._output_size,1))

    def forward(self, input):
        output1 = self._w1.dot(input.T)+self._b1
        output1 = relu(output1)

        output2 = self._w2.dot(output1) + self._b2
        out2_max = np.max(output2, axis=0)
        exponents = np.exp(output2-out2_max)
        output2 = exponents / exponents.sum(axis = 0)

        self._output1 = output1
        self._output2 = output2

    def calculate_dE(self, input, label):
        Y_U = label.T - self._output2
        self._dE2 = -Y_U.dot(self._output1.T)
        self._db2 = -Y_U.dot(np.ones((Y_U.shape[1], 1)))

        wY_U = (self._w2.T).dot(Y_U) * reluD(self._output1)
        self._dE1 = -wY_U.dot(input)
        self._db1 = -wY_U.dot(np.ones((wY_U.shape[1],1)))

    def backprop(self, learning_rate, size):
        self._w2 = self._w2 - learning_rate*self._dE2/size
        self._b2 = self._b2 - learning_rate*self._db2/size

        self._w1 = self._w1 - learning_rate*self._dE1/size
        self._b1 = self._b1 - learning_rate*self._db1/size

    def fit(self, input, label, validate_data = None, batch_size = 100, learning_rate = 0.1, epochs = 100):

        for epoch in range(epochs):
            for batch in range(input.shape[0]//batch_size):
                current_X = input[batch*batch_size:(batch+1)*batch_size]
                current_Y = label[batch*batch_size:(batch+1)*batch_size]
                self.forward(current_X)
                self.calculate_dE(current_X,current_Y)

                self.backprop(learning_rate, batch_size)


            prediction = self.predict(input)
            print("train acc: ", calculate_acc(label,prediction).round(4))

            if validate_data != None:

                prediction_val= self.predict(validate_data[0])
                print("validate acc: ", calculate_acc(validate_data[1],prediction_val).round(4))

    def predict(self, input):
        self.forward(input)
        return self._output2



