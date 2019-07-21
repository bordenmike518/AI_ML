import numpy as np

class Neuron:
    def __init__(self, address, dendrites=dict(), axonTerminals=list(), 
                 histLen=60, resting=0, threshold=1000, refactorPeriod=3, 
                 leak=25, inhibit=500):
        self.address        = address
        self.dendrites      = dendrites
        self.axonTerminals  = axonTerminals
        self.histLen        = histLen
        self.potential      = resting
        self.resting        = resting
        self.threshold      = threshold
        self.refactorPeriod = refactorPeriod
        self.leak           = leak
        self.inhibit        = inhibit
        self.history        = list()
        self.spike          = False
        self.stdpCount      = 0
        self.dwSum          = 0

    def STDP(self, address, lr, A, dt, ms):
        if (True): #(dt <= -2 or dt >= 2)):
            if (dt > 0):
                s, impact = -1, self.dendrites[address]
            else:
                s, impact =  1, 1 - self.dendrites[address]
            dw = lr * (s*A) * np.e ** (dt / (s*ms)) * impact
            # dw = lr * s*dt * impact
            self.dendrites[address] += dw
            #self.dwSum += dw if (dw > 0) else 0 #-dw


    def inputSpike(self, address):
        if (self.stdpCount < (self.histLen - self.refactorPeriod)):
            # Update potential, ...
            if (address not in self.dendrites.keys()):
                self.potential -= self.inhibit
            else:
                self.potential += self.dendrites[address]
                # ... history, ...
                self.history.append([address, 0])
                for i, [address, dt] in enumerate(self.history):
                    if(dt < -self.histLen):
                        del self.history[i]
                    else: break
                # ... and stdpCount.
                if (self.stdpCount > 0):
                    dt = self.histLen - self.stdpCount
                    self.STDP(address, 0.1, 0.6, dt, self.histLen) # Postsynaptic Potential

    def outputSpike(self):
        if (self.potential >= self.threshold):
            for address, dt in self.history:
                self.STDP(address, 0.1, 0.3, dt, self.histLen)      # Presynaptic Potential
            dhist = dict()
            for addr, w in self.history:
                if (addr not in dhist):
                    dhist[addr] = 1
                else:
                    dhist[addr] += 1
            mx = max(dhist.values()) * 0.5
            self.history = list()
            for k, v in dhist.items():
                if (v > mx):
                    self.history.append(k)
            acc, ace = 0,0
            for k, dv in self.dendrites.items():
                if (k not in self.history):
                     acc += dv
                else:
                     ace += dv
            for k, dv in self.dendrites.items():
                if (k not in self.history):
                    self.dendrites[k] = (300 - ace) * (dv / acc)
                else: 
                    self.dendrites[k] = dv 
            self.potential = self.resting
            self.stdpCount = self.histLen
            self.history = list()
            self.spike = True
            return self.spike

    def timeStep(self):
        if (not self.spike):
            df = self.potential - self.resting
            if (df != 0):
                self.potential -= self.leak if (df > self.leak) else df
            if (self.stdpCount > 0):
                self.stdpCount -= 1
            for i, _ in enumerate(self.history):
                self.history[i][1] -= 1
        #     self.threshold -= 0.1
        # else:
        #     self.threshold *= 1.2
        self.spike = False
