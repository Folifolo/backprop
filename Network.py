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

    def softmax(self, X):
        expX = np.exp(X)
        return expX / expX.sum(axis=0, keepdims=True)

    def forward(self, input):
        output1 = self._w1.dot(input.T)
        output2 = self._w2.dot(output1)

        #out2_max = np.max(output2, axis=0)
        #exponents = np.exp(output2-out2_max)
        #output2 = exponents / exponents.sum()
        output2 = self.softmax(output2)
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
        prediction = np.argmax(prediction, axis= 1)
        label = np.argmax(label, axis= 1)
        return (prediction == label).mean()

    def backprop(self, size):
        self._w2 = self._w2 - self._learning_rate*self._dE2/size
        self._w1 = self._w1 - self._learning_rate*self._dE1/size

    def fit(self, input, label, batch_size = 1, validate_data = None):
        #label = label[:100]
        prediction = np.zeros(label.shape)
        if validate_data != None:
            prediction_val = np.zeros(validate_data[1].shape)

        for epoch in range(self._epochs):
            for batch in range(batch_size):
                current_X = input[batch:batch+batch_size]
                current_Y = label[batch:batch+batch_size]
                self.forward(current_X)
                self.forward_dif(current_X)
                #print(str(np.argmax(self._output2))+" "+str(np.argmax(current_Y)))
                self.calculate_dE(current_X,current_Y)

                self.backprop(batch_size)

            for image in range(label.shape[0]):
                prediction[image] = self.predict(input[image])
                #print(self.calculate_E(current_Y,prediction, input.shape[0]))
            print("train acc: ", self.calculate_acc(label,prediction))

            if validate_data != None:
                for image in range(validate_data[1].shape[0]):
                    prediction_val[image] = self.predict(validate_data[0][image])
                    #print(self.calculate_E(current_Y,prediction, input.shape[0]))
                print("validate acc: ", self.calculate_acc(validate_data[1],prediction_val))

    def predict(self, input):
        self.forward(input)
        return self._output2

#if __name__ == '__main__':


