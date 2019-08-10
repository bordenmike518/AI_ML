import os
import re
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from contextlib import closing
import matplotlib.pyplot as plt
from multiprocessing import Pool
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import sent_tokenize, TweetTokenizer

class TextClassifier:
    def __init__(self, use_stemmer=True, remove_stopwords=True):
        self.use_stemmer, self.remove_stopwords = use_stemmer, remove_stopwords
        if (use_stemmer): 
            self.stemmer = SnowballStemmer('english')
        if (remove_stopwords): 
            self.stpwrds = set(stopwords.words('english'))
        self.filename = 'untitledModel'
        self.model = None
        self.word2vec = None
        self.features = list()
        self.categories = list()
#         self.embeddings = list()
        self.onehotlabels = list()
        self.indexFeatures = list()
        self.dictFeatures = {0: '<PAD>',
                             1: '<START>',
                             2: '<UNK>',
                             3: '<UNUSED>'}

    def loadData(self, text2d, labels=None, saveAs=None, 
                     use_stemmer=True, remove_stopwords=True):
        if(saveAs is not None): self.filename = saveAs
        if (type(text2d) is str):
            with open(DATA_DIR+text2d+'.pickle', 'rb') as file:
                data = pickle.load(file, encoding='latin1')
                try: self.model = keras.models.load_model(
                    DATA_DIR+self.filename+'.model')
                except: pass
                self.labels = data['labels']
                self.word2vec = data['word2vec']
#                 self.embeddings = data['embeddings']
                self.categories = data['categories']
                self.dictFeatures = data['dictFeatures']
                self.indexFeatures = data['indexFeatures']
        else:
            self.labels = labels
            self.categories = sorted(list(set(labels)))
            print('Running cleanAndTokenize')
            with closing(Pool(THREADS)) as p:
                self.features.extend(p.map(self.cleanAndTokenize,
                    text2d, chunksize=10))
            self.trainWord2Vec(self.features)
            self.indexAndEmedding()
            self.saveData({'labels': self.labels,
                           'word2vec': self.word2vec,
#                            'embeddings': self.embeddings,
                           'categories': self.categories,
                           'dictFeatures': self.dictFeatures,
                           'indexFeatures': self.indexFeatures}, 
                           self.filename)
            print('Saved!')

    def saveData(self, dictToSave, saveAs=None):
        print('Saving...')
        i, index = 1, ''
        if (saveAs is None):
            while(os.path.exists(DATA_DIR+self.filename+index+'.pickle')):
                index = '('+str(i)+')'; i += 1
        with open(DATA_DIR+saveAs+index+'.pickle', 'wb') as file:
            pickle.dump(dictToSave, file)

    def cleanAndTokenize(self, row):
        tknzr = TweetTokenizer()
        tmpRow = list()
        f = lambda x: x
        not_stopword = lambda x: True
        if (self.use_stemmer):
            f = lambda x: self.stemmer.stem(x)
        if (self.remove_stopwords):
            not_stopword = lambda x: x not in self.stpwrds
        replaceDict = {
            r"what's": "what is",
            r"\'s": "",
            r"\'ve": " have",
            r"can't": "cannot",
            r"n't": " not",
            r"i'm": "i am",
            r"\'re": " are",
            r"\'d": " would",
            r"\'ll": " will",
            r"\0s": "0",
            r"e-mail": "email"}
        for col in row:
            regex = re.compile("%s" % " ".join(map(re.escape, replaceDict.keys())))
            regex.sub(lambda s: replaceDict[mo.string[s.start():s.end()]], col.lower())
            tmpRow.extend([f(token.strip()) for token in tknzr.tokenize(col) 
                                       if not_stopword(token)])
        return tmpRow

    def trainWord2Vec(self, cleanToknizedSentences, iter=5, min_count=1, 
                      size=DIMS, window=5, workers=THREADS):
        print('Running trainWord2Vec')
        if (self.word2vec is None):
            self.word2vec = Word2Vec(cleanToknizedSentences, iter=iter, 
                                     min_count=min_count, size=size, 
                                     window=window, workers=workers)
        else:
            self.word2vec.train(cleanToknizedSentences, epochs=iter,
                                total_examples=len(cleanToknizedSentences))
        
    def indexAndEmedding(self):
        print('Running indexAndEmedding')
        getIndex = lambda x: self.word2vec.wv.vocab[x].index
        for row in self.features:
            drow = [1]
            for word in row:
                try:
                    i = getIndex(word)+3
                    drow.append(i)
                    self.dictFeatures[i] = word
                except:
                    drow.append(2)
            self.indexFeatures.append(drow)
#         del self.features
#         self.embeddings = np.zeros((len(self.word2vec.wv.vocab), DIMS))
#         for i in range(len(self.word2vec.wv.vocab)):
#             buff = self.word2vec.wv[self.word2vec.wv.index2word[i]]
#             if (buff is not None):
#                 self.embeddings[i] = buff

    def decodeIndexFeatures(self, indexFeatures):
        return ' '.join([self.dictFeatures[i] for i in indexFeatures])

    def trainModel(self, tvt=[0.6, 0.25, 0.15], saveModel=False, epochs=5):
        print('Running trainModel')
        assert sum(tvt) > 0.999 and sum(tvt) < 1.001, "tvt should sum to 1"
        sz = len(self.indexFeatures)
        maxlen = len(max(self.indexFeatures, key=lambda x: len(x)))
        indexFeatures = keras.preprocessing.sequence.pad_sequences(
            self.indexFeatures, value=0, padding='post', maxlen=4096)
        xtrain, xvalidation, xtest = np.split(indexFeatures, 
                                     [int(sz*tvt[0]), int(sz*sum(tvt[:2]))])
        ytrain, yvalidation, ytest = np.split(list(map(
                                     self.categories.index, self.labels)), 
                                     [int(sz*tvt[0]), int(sz*sum(tvt[:2]))])
        if (self.model is None):
            self.model = keras.Sequential([
                keras.layers.Embedding(self.word2vec.wv.vectors.shape[0], 
                                       DIMS, input_length=4096),
                keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)),
                keras.layers.Bidirectional(keras.layers.LSTM(128)),
                keras.layers.Dense(256, activation='tanh'),
                keras.layers.Dense(128, activation='tanh'),
                keras.layers.Dense(len(self.categories), activation='softmax')])
        self.model.summary()
        self.model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.history = self.model.fit(xtrain, ytrain, epochs=epochs, batch_size=64,
                                      validation_data=(xvalidation, yvalidation))
        loss, acc = self.model.evaluate(xtest, ytest)
        if (saveModel):
            keras.models.save_model(self.model, DATA_DIR+self.filename+'.model')
        print('loss = {}\nacc = {}'.format(loss, acc))

    def to_onehot(self, labels):
        onehotlabels = keras.utils.to_categorical(
                             np.array(list(map(
                                 self.categories.index, 
                                 labels))).reshape(len(labels),1), 
                             num_classes=len(self.categories))
        return onehotlabels
