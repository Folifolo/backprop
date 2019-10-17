import numpy as np

class NN:
    def __init__(self, input_size = 784, hidden_size=30, output_size=10, learning_rate = 0.0001, epochs = 100):
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size

        self._learning_rate = learning_rate
        self._epochs = epochs

        self._w1 = np.array([])
        self._w2 = np.array([])

        self._output1 = np.array([])
        self._output2 = np.array([])

        self._div1 = np.array([])
        self._div2 = np.array([])

        self._dE1 = np.array([])
        self._dE2 = np.array([])

    def init_weights(self):
        self._w1 =np.random.randn(self._hidden_size, self._input_size)/10
        self._w2 =np.random.randn(self._output_size, self._hidden_size)/10

    def forward(self, input):
        output1 = self._w1.dot(input.T)
        output2 = self._w2.dot(output1)

        out2_max = max(output2)
        exponents = np.exp(output2-out2_max)
        output2 = exponents / exponents.sum()

        self._output1 = output1
        self._output2 = output2


    def forward_dif(self, input):
        self._div1 = np.ones(self._output1.shape)
        self._div2 = self._output2*(1-self._output2)


    def calculate_dE(self, input, label):
        delta2 = (label.T-self._output2)
        self._dE2 = -(delta2).dot(self._output1.T)

        delta1 = self._w2.T.dot(delta2)
        self._dE1 = -delta1.dot(input)

    def calculate_E(self, label, prediction, size):
        return -np.sum(label * np.log(prediction.T)) / size


    def calculate_acc(self, label, prediction):
        prediction = np.argmax(prediction)
        label = np.argmax(label)
        return (prediction == label).mean()

    def backprop(self, size):
        self._w2 = self._w2 - self._learning_rate*self._dE2/size
        self._w1 = self._w1 - self._learning_rate*self._dE1/size

    def fit(self, input, label, batch_size = 1):
        label = label[:100]
        prediction = np.zeros(label.shape)
        for epoch in range(self._epochs):
            for image in range(label.shape[0]):
                currend_X = input[image:image+batch_size]
                currend_Y = label[image:image+batch_size]
                self.forward(currend_X)
                self.forward_dif(currend_X)

                self.calculate_dE(currend_X,currend_Y)

                self.backprop(label.shape[0])

            for image in range(label.shape[0]):
                prediction[image] = self.predict(input[image])
                #print(self.calculate_E(currend_Y,prediction, input.shape[0]))
            print(self.calculate_acc(label,prediction))

    def predict(self, input):
        self.forward(input)
        return self._output2

#if __name__ == '__main__':


