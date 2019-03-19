import numpy as np
import time
np.random.seed(int(time.time()))

class NeuronLayer:
    def __init__(self, prevCount, count, first=False):
        self.prevCount = prevCount
        self.count = count
        self.neurons = None
        self.errors = None
        self.bias = np.ones((self.count, 1))
        if(not first):
            self.synapses = np.random.rand(self.count, self.prevCount)*2-1

class NeuralNetwork:
    def __init__(self):
        self.layers = 0
        self.network = list()

    def initNetwork(self, networkStruct):
        self.layers = len(networkStruct)
        self.network.append(NeuronLayer(0, networkStruct[0], True))
        for i in range(1, self.layers):
            self.network.append(NeuronLayer(networkStruct[i-1], networkStruct[i]))

    def feedForward(self, data):
        self.network[0].neurons = data.reshape(self.network[0].count, 1)
        for i in range(1, self.layers-1):
            self.network[i].neurons = self.tanh(np.dot(self.network[i].synapses, 
                                                         self.network[i-1].neurons))
        self.network[i+1].neurons = self.softmax(np.dot(self.network[i+1].synapses, 
                                                         self.network[i].neurons))
        

    def backpropagation(self, target, learningRate):
        self.network[-1].errors = 2 * (target-self.network[-1].neurons) * \
                                  self.tanh_derivative(self.network[-1].neurons)
        for i in reversed(range(1, self.layers)):
            self.network[i].synapses += learningRate * np.dot(
                self.network[i].errors,
                self.network[i-1].neurons.T
            )
            self.network[i-1].errors = 2 * np.dot(
                self.network[i].synapses.T,
                self.network[i].errors
            ) * self.tanh_derivative(self.network[i-1].neurons)

    def train(self, trainLabels, trainData, epochs=1, testLabels=[], 
                    testData=[], learningRate=None, withOutput=False):
        for i in range(epochs):
            print('\t-- Epoch {}'.format(i+1))
            for label, data in zip(trainLabels, trainData):
                if(type(label) == int):
                    target = self.oneHotEncode(label)
                else:
                    target = label
                self.feedForward(data)
                self.backpropagation(target, learningRate)
            if(withOutput):
                accuracy = self.test(testLabels, testData)
                print('Accuracy = {0:.2f}%'.format(accuracy*100))

    def test(self, labels, testData):
        correct = 0
        for i, (label, data) in enumerate(zip(labels, testData)):
            self.feedForward(data)
            bestIndex = np.argmax(self.network[-1].neurons)
            if (type(label) == int):
                if (label == bestIndex):
                    correct += 1
            else:
                if (np.argmax(label) == bestIndex):
                    correct += 1
        return correct/len(labels)

    def oneHotEncode(self, index):
        vect = np.zeros((self.network[-1].count, 1))
        vect[index][0] = 1
        return vect
    
    def logLikelihood(self, y, yhat):
        return np.where((y/2)-0.5+yhat > 0, -np.log(np.abs((y/2)-0.5+yhat)), 0)

    def step(self, A):
        return np.where(A > 0, 1, 0)

    def sigmoid(self, A):
        return 1 / (1 + np.exp(-A))

    def sigmoid_derivative(self, A):
        return A * (1 - A)

    def tanh(self, A):
        return np.tanh(A)
    
    def tanh_derivative(self, A):
        return 1 - np.power(np.tanh(A), 2)

    def ReLU(self, A):
        return np.where(A > 0, A, 0)

    def softmax(self, A):
        e = np.exp(A - np.max(A))
        return e / e.sum()
    
    def entropy(p):
        return -np.sum(p*np.log2(p))
        
    def crossEntropy(self, p, q):
        '''
        p = true probability distribution (expected)
        q = predicted probability distribution (guessed)
        '''
        return -np.sum(np.where(q > 0, p*np.log2(q), 0))

    def KLDivergence(self, p, q):
        return self.crossEntropy(p,q) - self.entropy(p)

class DataLoader:
    def __init__(self):
        pass

    def standardize(self, A):
        return (A - np.mean(A)) / np.std(A)

    def normalize(self, A):
        return (A - np.min(A)) / (np.max(A) - np.min(A))

    def extractMNIST(self, fileName):
        labels = []
        fname = open(fileName, "r")
        values = fname.readlines()[:20000]
        fname.close()
        for i, record in enumerate(values):
            data = record.split(",")
            values[i] = self.standardize(np.asfarray(data[1:]))
            labels.append(int(data[0]))
        return labels, values

def main():
    # Test input parameters
    network = [784, 200, 10]
    epochs = 10
    learningRate = 0.0019
    displayOutput = True
    dl = DataLoader()

    # Create neural network
    print("Creating Network")
    ann = NeuralNetwork()
    ann.initNetwork(network)

    # Open file to loop through
    print("Opening Training Data")
    MNIST_Train_Labels, MNIST_Train_Values = dl.extractMNIST("MNIST/mnist_train.csv")
    print("Opening Testing Data")
    MNIST_Test_Labels, MNIST_Test_Values = dl.extractMNIST("MNIST/mnist_test.csv")

    # Train
    print("Training:")
    ann.train(MNIST_Train_Labels, MNIST_Train_Values, epochs,
              MNIST_Test_Labels, MNIST_Test_Values, learningRate, displayOutput)

    # Test
    if (not displayOutput):
        print("Testing:")
        accuracy = ann.test(MNIST_Test_Labels, MNIST_Test_Values)

        # Print Accuracy
        print("Accuracy = %.2f%%" % (accuracy * 100))

if __name__ == '__main__':
    main()
'''
1 error = 2*(self.y - self.output) * sigmoid_derivative(self.output)
2 d_weights2 = np.dot(self.layer1.T, (error))
3 d_weights1 = np.dot(self.input.T,  (np.dot(error, self.weights2.T) * sigmoid_derivative(self.layer1)))
'''











