import os
import numpy as np
from time import time, sleep
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
from Neuron import Neuron
from spikeTrains import getMNISTspikes

np.random.seed(int(time()))

class Brain:
    def __init__(self, name='DefaultBrain'):
        self.name    = name
        if (os.path.exists('{}'.format(self.name))):
            if (os.path.exists('{}/{}.pickle'.format(self.name, self.name))):
                print('Reloading previous brain!')
                f = open(self.name+'/'+self.name+'.pickle', 'rb')
                pickleDict = pickle.load(f)
                self.input   = pickleDict['input']
                self.network = pickleDict['network']
                self.stats   = pickleDict['stats']
                self.labels  = pickleDict['labels']
                self.inXY    = pickleDict['inXY']
                self.netXY   = pickleDict['netXY']
                self.lasti   = pickleDict['lasti']
                return
        else:
            os.mkdir(self.name)
            os.mkdir(self.name+'/images/')
        print('Starting a fresh baby brain!')
        self.input   = dict()
        self.network = dict()
        self.stats   = dict()
        self.labels  = dict()
        self.inXY    = None
        self.netXY   = None
        self.lasti   = 0

    def imageTopology(self, inXY, netXY):
        if (not self.inXY):
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

    def stimulate(self, labels, spikeTrains, bins):
        fig, axs = plt.subplots(self.netXY[0], self.netXY[1])
        self.stats = {'01,{0:02d},{1:02d}'.format(j, i): {str(k): 0 for k in range(10)} for j in range(self.netXY[0]) for i in range(self.netXY[1])}
        self.bins = bins
        order = list(range(len(spikeTrains)))
        np.random.shuffle(order)
        self.zrange = list(range(self.inXY[0]))
        self.yrange = list(range(self.inXY[1]))
        self.jrange = list(range(self.netXY[0]))
        self.irange = list(range(self.netXY[1]))
        # For each sample...
        for index, w in enumerate(order):
            index += self.lasti
            # For each time step...
            for x in range(self.bins):
                np.random.shuffle(self.zrange)
                # For each input neuron row...
                for z in self.zrange:
                    np.random.shuffle(self.yrange)
                    # For each input neuron column...
                    for y in self.yrange:
                        # If the neuron has spiked...
                        if (spikeTrains[w][z][y][x] == 1):
                            spikeAddr = '00,{0:02d},{1:02d}'.format(z, y)
                            np.random.shuffle(self.input[spikeAddr].axonTerminals)
                            # For each excitatory neuron's axon terminals... 
                            for exciteAddr in self.input[spikeAddr].axonTerminals:
                                self.network[exciteAddr].inputSpike(spikeAddr)
                                # If the network layer neuron spikes...
                                if (self.network[exciteAddr].outputSpike()):
                                    self.stats[exciteAddr][str(labels[w])] += 1
                                    np.random.shuffle(self.network[exciteAddr].axonTerminals)
                                    # Inhibit all lateral neurons in the network layer.
                                    for inhibitAddr in self.network[exciteAddr].axonTerminals:
                                        self.network[inhibitAddr].inputSpike(exciteAddr)
                np.random.shuffle(self.jrange)
                # For each network neuron row...
                for j in self.jrange:
                    np.random.shuffle(self.irange)
                    # For each network neuron column...
                    for i in self.irange:
                        netAddr = '01,{0:02d},{1:02d}'.format(j, i)
                        self.network[netAddr].timeStep()
            print('Image: {}: {}'.format(index+1, labels[w]))
            # Imaging
            for address, neuron in self.network.items():
                _, y1, x1 = list(map(int, address.split(',')))
                scan = np.array(sorted(neuron.dendrites.items(), key=lambda item: item[0]))[:,1].astype(np.float).reshape(28,28)
                axs[y1, x1].cla()
                axs[y1, x1].imshow(scan, cmap=cm.get_cmap('YlGn'),
                    aspect="auto", vmin=0,vmax=1)
                axs[y1, x1].set_title('{}'.format(address))
                axs[y1, x1].relim()
            fig.suptitle('Image {}: {}'.format(index+1, labels[w]), fontsize=16)
            fig.savefig('{}/images/Image_{}.png'.format(self.name, index))
            plt.pause(0.05)
        self.lasti = index
        self.labels = dict()
        for spikeAddr, spikeCounts in self.stats.items():
            for digit, count in spikeCounts.items():
                if (spikeAddr in self.labels):
                    if (count < self.labels[spikeAddr][1]):
                        self.labels[spikeAddr] = [digit, count]
                else:
                    self.labels[spikeAddr] = [digit, count]
        self.saveBrain()

    def saveBrain(self, overrideIfExists=True):
        pickleData =   {'input'  : self.input,
                        'network': self.network,
                        'stats'  : self.stats,
                        'labels' : self.labels,
                        'inXY'   : self.inXY,
                        'netXY'  : self.netXY,
                        'lasti'  : self.lasti}
        f = open('{}/{}.pickle'.format(self.name, self.name), 'wb')
        pickle.dump(pickleData, f)
        f.close()

    def test(self, label, spikeTrain):
        votes = []
        if (not self.zrange):
            self.zrange = list(range(self.inXY[0]))
            self.yrange = list(range(self.inXY[1]))
            self.jrange = list(range(self.netXY[0]))
            self.irange = list(range(self.netXY[1]))
        for x in range(self.bins):
            np.random.shuffle(self.zrange)
            # For each input neuron row...
            for z in self.zrange:
                np.random.shuffle(self.yrange)
                # For each input neuron column...
                for y in self.yrange:
                    # If the neuron has spiked...
                    if (spikeTrain[z][y][x] == 1):
                        spikeAddr = '00,{0:02d},{1:02d}'.format(z, y)
                        np.random.shuffle(self.input[spikeAddr].axonTerminals)
                        # For each excitatory neuron's axon terminals... 
                        for exciteAddr in self.input[spikeAddr].axonTerminals:
                            self.network[exciteAddr].inputSpike(spikeAddr)
                            # If the network layer neuron spikes...
                            if (self.network[exciteAddr].outputSpike()):
                                votes.append(self.labels[exciteAddr][0])
            np.random.shuffle(self.jrange)
            # For each network neuron row...
            for j in self.jrange:
                np.random.shuffle(self.irange)
                # For each network neuron column...
                for i in self.irange:
                    netAddr = '01,{0:02d},{1:02d}'.format(j, i)
                    self.network[netAddr].timeStep()
        if (len(votes) > 0):
            popVote = max(set(votes), key=votes.count)
        else:
            popVote = 'X'
        print('{} : {} : {}'.format(popVote, label, popVote == str(label)))
        return True if(popVote == label) else False


def normalize(X):
    mn = np.min(X)
    return (X - mn) / (np.max(X) - mn)

def  standardize(X):
    return (X - np.mean(X)) / np.std(X)

def main():

    directory = "/home/michael/Documents/workspace/python/artificial_intelligence/datasets/MNIST/"
    bins = 1000

    print('Loading spike trains...')
    labels, spikeTrains = getMNISTspikes(directory + 'mnist_train.csv', bins)
    split = int(len(labels) * 0.7)
    trainLabels, testLabels = labels[:split], labels[split:]
    trainSpikeTrains, testSpikeTrains = spikeTrains[:split], spikeTrains[split:]

    print('Setting up the brain...')
    brain = Brain('MNIST_Brain')
    if (not brain.inXY):
        brain.imageTopology([28, 28], [7, 7])

    # print(len(brain.network['01,00,00'].dendrites))
    print('Simulating...')
    brain.stimulate(trainLabels, trainSpikeTrains, bins)

    correct = 0
    for label, spikeTrain in zip(testLabels, testSpikeTrains):
        if (brain.test(label, spikeTrain)):
            correct += 1
    print('Accuracy = {0:.2f}'.format(correct/len(labels)))



if __name__ == '__main__':
    main()
