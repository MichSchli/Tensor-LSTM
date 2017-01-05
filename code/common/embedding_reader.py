import numpy as np
import sys
import pickle

"""
Abstract class reading word embeddings from differently formatted files.
"""
class EmbeddingReader():

    always_lower = False
    dim = None
    d = {}
    
    def __getitem__(self, item):
        if self.always_lower:
            item = item.lower()
                
        if item in self.d:
            return self.d[item]
        else:
            return np.zeros(self.dim, dtype=np.float32)


"""
Glove format:
"""
class GloveReader(EmbeddingReader):

    glove_path = 'misc/glove.6B/glove.6B.50d.txt'
    dim=50
    
    def __init__(self):
        print("Loading GLOVE-dataset...", file=sys.stderr)
        for line in open(self.glove_path):
            parts = line.strip().split(' ')

            self.d[parts[0]] = np.array([float(x) for x in parts[1:]])


"""
Polyglot Format:
"""
class PolyglotReader(EmbeddingReader):

    dim=64
    
    def __init__(self, language='en'):
        print("Loading polyglot...", file=sys.stderr)
        polyglot_path = 'misc/Polyglot/polyglot-' + language + '.pkl'

        f = open(polyglot_path, 'rb')
        words, vecs = pickle.load(f, encoding='latin1')

        self.d = dict(zip(words, vecs))


"""
Multilingual embeddings following the format of Levy et al., 2016.
"""     
class MultilingualEmbeddingReader(EmbeddingReader):

    embedding_path = 'misc/bible_multi-sid-pmi_submitted.vecs'
    dim=500
    
    always_lower = True

    def __init__(self, language=None):
        print("Loading embeddings...", file=sys.stderr)
        
        f = open(self.embedding_path, 'r')
        for line in f:
            parts = line.split(' ')
            word = parts[0]
            if language is not None:
                if parts[0][-2:] != language:
                    continue
                else:
                    word = parts[0][:-3]

            self.d[word] = np.array([float(x) for x in parts[1:]])
            
