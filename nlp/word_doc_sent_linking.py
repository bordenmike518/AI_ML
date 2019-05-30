import operator
from multiprocessing import Pool
from pathlib2 import Path
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, TweetTokenizer

# Helper class
class Word:
    # First 2 args are important to user. The word, document name, and 
    # the sentence the word is in.
    def __init__(self, word, document, sentenceIndex=None):
        self.word = word
        self.count = 0
        if (type(document) == dict):
            self.documents = document
        else:
            self.documents = dict()
            self.update(document, sentenceIndex)
    
    # After the word has been intitialized, continue to update it 
    # with more documents and the sentences they are found in.
    def update(self, document, sentenceIndex):
        self.count += 1
        if (document not in self.documents):
            self.documents[document] = {sentenceIndex}
        else:
            self.documents[document].add(sentenceIndex)

# Main class to be used
class Lexicon:
    # Can be initiailized with information from previous lexicons.
    def __init__(self, name, lexicon=dict(), documents=dict()):
        self.name = name
        self.lexicon = lexicon
        self.documents = documents
        self.stopwords = set(stopwords.words('english'))
        self.maxPoolSize = 8 # **** UPDATE WITH MAX CORES TO BE USED ******
    
    # Updates the lexicon. If a file or directory containing files for updating.
    def update(self, path):
        if (type(path) != str):
            index = path[1]
            path = path[0]
        config = Path(path)
        tknzr = TweetTokenizer()
        lex, doc = dict(), dict()
        if (config.is_file()):  # If just a file
            with config.open() as f:
                sentences = sent_tokenize(f.read())
                document = path.split('/')[-1]
                doc[document] = list(sentences)
                for i, sentence in enumerate(sentences):
                    token_list = tknzr.tokenize(sentence)
                    for word in token_list:
                        if (word not in self.stopwords and
                            self.is_not_number(word) and 
                            len(word) > 1):
                            if (pos_tag([word])[0][1] != 'NNP'): word = word.lower()
                            if (word not in lex):
                                lex[word] = Word(word, document, i)
                            else:
                                lex[word].update(document, i)
            return (lex, doc)
        elif (config.is_dir()): # If directory, recursively read each directory and file
            if (path[-1] != '/'): path = path + '/'
            iterdir = list()
            for d in config.iterdir():
                iterdir.append(path+d.name)
            self.outputs = [0] * len(list(iterdir))
            p = Pool(self.maxPoolSize)
            outputs = p.map(self.update, zip(iterdir, list(range(len(iterdir)))))
            for (lex, doc) in outputs:
                for k, v in lex.items():
                    if (k not in self.lexicon):
                        self.lexicon[k] = Word(k, v.documents)
                    self.lexicon[k].count += v.count
                    self.lexicon[k].documents.update(v.documents)
                self.documents.update(doc)
        else:
            print('{} doesn\'t exist.'.format(path))

    # Shows the top N words in the lexicon
    def showTopN(self, N, show=False):
        for word in sorted(self.lexicon.values(), key=operator.attrgetter('count'), reverse=True)[:N]:
            print('''{} >>>
    Count     : {}
    Documents : {}'''.format(word.word, word.count, len(list(word.documents.keys()))))
            count = 0
            for k, vl in word.documents.items():
                for v in vl:
                    if (show):
                        print('    ' + self.documents[k][v])
                    else:
                        count += 1
            if (show):
                print('    Sentences : {}'.format(count))

    # Used to check if the word is a number of 
    def is_not_number(self, token):
        return any(c.isdigit for c in token)

def main():
    # Initialize the Lexicon class and give it a name
    lex = Lexicon('Test')
    # Give it a file or a directory full of files and it will go 
    # through the file[s] one at a time, updating the lexicon.
    lex.update('./test_docs')
    # Print out the top N words in the format
    # word >>>
    #     Count     : How many times it appeared.
    #     Documents : How many documents it appeared in.
    #     Sentences : How many sentences it appeared in.
    #
    # change to 'lex.showTopN(10, True)' to see all sentences
    lex.showTopN(10)

if __name__ == '__main__': 
    main()
