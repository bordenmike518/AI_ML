import numpy as np
from scipy.special  import factorial
import matplotlib.pyplot as plt

def pois(x, m):
    return np.power(m, x) * np.exp(-m) / factorial(x)

def diff(spiketrain):
    dlist = list()
    count = 1
    for spike in spiketrain:
        if(spike == 1):
            dlist.append(count)
            count = 1
        else:
            count += 1
    return dlist

def getCounts(X):
    counts = dict()
    for x in X:
        if (x not in counts):
            counts[x] = 1
        else:
            counts[x] += 1
    return counts

def poissonSpikeTrain(spikes, totaltime):
    spiketrain = np.zeros(totaltime, dtype=np.int32)
    spikes = 1 if (spikes == 0) else spikes
    rate = int(totaltime / spikes + 1)
    bins = int((totaltime / rate) + 1)
    isi = np.cumsum(np.random.poisson(rate, bins))
    i = 0
    for s in isi: 
        if (s >= totaltime-rate): break
        else: i += 1
    spiketrain[isi[:i]] = 1
    return isi[:i], spiketrain

def uniformSpikeTrain(prob, totaltime):
    spiketrain = np.random.rand(int(totaltime/2))
    spiketrain = [1 if(spike < prob) else 0 for spike in spiketrain]
    spiketrain2 = np.random.rand(int(totaltime/2))
    spiketrain2 = [1 if(spike < 0.05) else 0 for spike in spiketrain]
    spiketrain.extend(spiketrain2)
    isi, dt, runtotal = list(), 1, 0
    for spike in spiketrain:
        if (spike == 0):
            dt += 1
        else:
            runtotal += dt
            isi.append(runtotal)
            dt = 1
    return isi, spiketrain

def getMNISTspikes(fileName):
    fdata = open(fileName, "r")
    images = fdata.readlines()
    fdata.close()
    labels = list()
    spikeTrains = list()
    # data = dict()
    for image in images[:100]:
        pixels = list(map(int, image.split(',')))
        labels.append(int(pixels[0]))
        pixels.pop(0)
        imageSpikeTrains = list()
        for y in range(28):
            rowSpikeTrains = list()
            for x in range(28):
                pixelProb = pixels[(y*28)+x]/255
                prob = 0.1 if (pixelProb < np.random.randint(1,6)/100) else pixelProb
                _, spiketrain = poissonSpikeTrain(int(prob*100), 100)
                rowSpikeTrains.append(spiketrain)
            imageSpikeTrains.append(rowSpikeTrains)
        spikeTrains.append(imageSpikeTrains)
    #     if (labels[-1] not in data):
    #         data[labels[-1]] = [np.sum(imageSpikeTrains), 1]
    #     else:
    #         data[labels[-1]][0] += np.sum(imageSpikeTrains)
    #         data[labels[-1]][1] += 1
    # for k,v in data.items():
    #     print('{}: {}'.format(k, v[0]/v[1]))
    return labels, spikeTrains

'''
0: 1359.4156677359447
1: 595.7680213586473
2: 1167.8799932863376
3: 1109.948295547219
4: 951.371619308456
5: 1009.6511713705959
6: 1076.6241973639744
7: 897.8207501995212
8: 1177.3137925141002
9: 960.8322407127248
'''

# isi = list()
# spi = list()
# for i in range(1,101):
#     st = uniformSpikeTrain(0.75, 1000)
#     isi.append(st[0])
#     spi.append(st[1])
# # plt.eventplot(isi)
# plt.plot(np.arange(1000), np.sum(spi, axis=0))
# plt.show()