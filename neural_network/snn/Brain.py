import numpy as np
from time import time, sleep
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Neuron import Neuron
from spikeTrains import getMNISTspikes
from pandas import DataFrame

np.random.seed(int(time()))

labels = None

class Brain:
    def __init__(self):
        self.input   = dict()
        self.network = dict()
        self.inXY    = None
        self.netXY   = None

    def imageTopology(self, inXY, netXY):
        self.inXY, self.netXY = inXY, netXY
        inAxonTerminals = ['01,{0:02d},{1:02d}'.format(j,i) for j in range(netXY[0]) for i in range(netXY[1])]
        for j in range(inXY[0]):    # Input Layer (L0)
            for i in range(inXY[1]):
                address = '00,{0:02d},{1:02d}'.format(j,i)
                self.input[address] = Neuron(address, dict(), inAxonTerminals)
        for j in range(netXY[0]):    # Excitatory and Inhibitory Layer (L1)
            for i in range(netXY[1]):
                address = '01,{0:02d},{1:02d}'.format(j,i)
                netDendrites = {'00,{0:02d},{1:02d}'.format(j,i): np.random.uniform(0.3, 0.7) for j in range(inXY[0]) for i in range(inXY[1])}
                netAxonTerm = list(inAxonTerminals)
                netAxonTerm.remove(address)
                self.network[address] = Neuron(address, netDendrites, netAxonTerm)

    def spikeTrain(self, probability, dtCount):
        return [1 if(x < probability) else 0 for x in np.random.rand(dtCount)]

    def stimulate(self, spikeTrains, labels):
        fig, axs = plt.subplots(self.netXY[0], self.netXY[1])
        stats = {str(k): {'01,{0:02d},{1:02d}'.format(j, i): 0 for j in range(self.netXY[0]) for i in range(self.netXY[1])} for k in range(10)}
        order = list(range(len(spikeTrains)))
        np.random.shuffle(order)
        for w, index in enumerate(order):
            image = spikeTrains[index]
            for x in range(100):
                zrange = list(range(self.inXY[0]))
                np.random.shuffle(zrange)
                for z in zrange:
                    yrange = list(range(self.inXY[1]))
                    np.random.shuffle(yrange)
                    for y in yrange:
                        # print (image[z][y][x])
                        if (image[z][y][x] == 1):
                            spikeAddr = '00,{0:02d},{1:02d}'.format(z, y)
                            np.random.shuffle(self.input[spikeAddr].axonTerminals)
                            for exciteAddr in self.input[spikeAddr].axonTerminals:
                                self.network[exciteAddr].inputSpike(spikeAddr)
                                if (self.network[exciteAddr].outputSpike()):
                                    stats[str(labels[index])][exciteAddr] += 1
                                    np.random.shuffle(self.network[exciteAddr].axonTerminals)
                                    for inhibitAddr in self.network[exciteAddr].axonTerminals:
                                        self.network[inhibitAddr].inputSpike(exciteAddr)
                jrange = list(range(self.inXY[0]))
                np.random.shuffle(jrange)
                for j in range(self.netXY[0]):
                    irange = list(range(self.inXY[0]))
                    np.random.shuffle(irange)
                    for i in range(self.netXY[1]):
                        netAddr = '01,{0:02d},{1:02d}'.format(j, i)
                        self.network[netAddr].timeStep()
            print('Image: {}: {}'.format(w, labels[index]))
            for addr, neuron in self.network.items():
            	neuron.spikeCount = 0
            #    print('{}: {}'.format(addr, neuron.dwSum))
            for k, v in stats[str(labels[index])].items():
                print('{} : {}'.format(k, v))
            # Imaging
            for address, neuron in self.network.items():
                _, y1, x1 = list(map(int, address.split(',')))
                scan = np.array(sorted(neuron.dendrites.items(), key=lambda item: item[0]))[:,1].astype(np.float).reshape(28,28)
                axs[y1, x1].cla()
                axs[y1, x1].imshow(scan, cmap=cm.get_cmap('YlGn'),
                    aspect="auto", vmin=0,vmax=1)
                axs[y1, x1].set_title('{}'.format(address))
                axs[y1, x1].relim()
            fig.suptitle('Image {}: {}'.format(w, labels[index]))
            plt.pause(0.05)

def normalize(X):
    mn = np.min(X)
    return (X - mn) / (np.max(X) - mn)

def  standardize(X):
    return (X - np.mean(X)) / np.std(X)

def main():

    directory = "/home/michael/Documents/workspace/python/artificial_intelligence/datasets/MNIST/"

    print('Loading spike trains...')
    labels, spikeTrains = getMNISTspikes(directory + 'mnist_train.csv')

    print('Setting up the brain...')
    brain = Brain()
    brain.imageTopology([28, 28], [10, 30])

    # print(len(brain.network['01,00,00'].dendrites))
    print('Simulating...')
    brain.stimulate(spikeTrains, labels)



if __name__ == '__main__':
    main()
