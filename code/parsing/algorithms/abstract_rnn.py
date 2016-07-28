import numpy as np
from theano import tensor as T
import theano
import pickle
import os.path
import imp
import sys

sys.setrecursionlimit(10000)

optimizers = imp.load_source('optimizers', 'code/parsing/algorithms/optimizers.py')
io = imp.load_source('io', 'code/common/io.py')

class RNN():

    predict_graph = None
    optimizer = None
    
    def __init__(self, feature_mode, optimizer_config_path):
        self.feature_mode = feature_mode
        if optimizer_config_path is not None:
            self.optimizer = optimizers.from_config(io.read_config_file(optimizer_config_path))

            self.optimizer.set_initial_weights(self.get_weight_list())
            self.optimizer.set_loss_function(self.build_loss_graph())
            self.optimizer.set_gradient_function(self.build_single_gradient_graph())
            self.optimizer.set_update_function(self.update_function)

            self.optimizer.initialize()

    '''
    Update:
    '''
    def update_function(self, weights):
        self.update_weights(weights)
        self.save(self.save_path)

    '''
    Weight functions:
    '''
    
    def get_weight_list(self):
        return [weight for layer in self.layers for weight in layer.get_python_weights()]

    
    def get_theano_weight_list(self):
        return [weight for layer in self.layers for weight in layer.get_theano_weights()]

    
    def update_weights(self, update_list):
        prev_count = 0
        for layer in self.layers:
            current_count = prev_count + layer.weight_count()
            layer.update_weights(update_list[prev_count:current_count])
            prev_count = current_count

    
    '''
    Prediction
    '''
    
    def build_predict_graph(self, saved_graph=None):
       
        print("Building prediction graph...")

        for l in self.layers:
            l.set_training(False)
        
        Sentence = T.fmatrix('Sentence')
        Characters = T.ftensor3('Characters')
        WordLengths = T.ivector('WordLengths')
        
        weight_list = self.get_theano_weight_list()

        if self.feature_mode == 'character':
            result = self.theano_sentence_prediction(Characters, WordLengths)
            input_list = [Characters, WordLengths] + list(weight_list)
        elif self.feature_mode == 'sentence':
            result = self.theano_sentence_prediction(Sentence)
            input_list = [Sentence] + list(weight_list)
        elif self.feature_mode == 'both':
            result = self.theano_sentence_prediction(Sentence, Characters, WordLengths)
            input_list = [Sentence, Characters, WordLengths] + list(weight_list)

        cgraph = theano.function(inputs=input_list, outputs=result, mode='FAST_RUN', allow_input_downcast=True)

        print("Done building graph.")

        return cgraph
    
            
    '''
    Loss:
    '''
    
    def build_loss_graph(self, saved_graph=None):
        print("Building loss graph...")

        for l in self.layers:
            l.set_training(False)

        Sentence = T.fmatrix('Sentence')
        Characters = T.ftensor3('Characters')
        WordLengths = T.ivector('WordLengths')
        GoldPredictions = T.fmatrix('GoldPredictions')
        
        weight_list = self.get_theano_weight_list()

        if self.feature_mode == 'character':
            result = self.theano_sentence_loss(Characters, WordLengths, GoldPredictions)
            input_list = [Characters, WordLengths, GoldPredictions] + list(weight_list)
        elif self.feature_mode == 'sentence':
            result = self.theano_sentence_loss(Sentence, GoldPredictions)
            input_list = [Sentence, GoldPredictions] + list(weight_list)
        elif self.feature_mode == 'both':
            result = self.theano_sentence_loss(Sentence, Characters, WordLengths, GoldPredictions)
            input_list = [Sentence, Characters, WordLengths, GoldPredictions] + list(weight_list)

        cgraph = theano.function(inputs=input_list, outputs=result, mode='FAST_RUN', allow_input_downcast=True)

        print("Done building graph.")
        
        return cgraph

            
    '''
    SGD:
    '''

    def build_single_gradient_graph(self):
        print("Building gradient graph...")

        for l in self.layers:
            l.set_training(True)

        Sentence = T.fmatrix('Sentence')
        Characters = T.ftensor3('Characters')
        WordLengths = T.ivector('WordLengths')
        GoldPredictions = T.fmatrix('GoldPredictions')
        
        weight_list = self.get_theano_weight_list()

        if self.feature_mode == 'character':
            loss = self.theano_sentence_loss(Characters, WordLengths, GoldPredictions)
            input_list = [Characters, WordLengths, GoldPredictions] + list(weight_list)
        elif self.feature_mode == 'sentence':
            loss = self.theano_sentence_loss(Sentence, GoldPredictions)
            input_list = [Sentence, GoldPredictions] + list(weight_list)
        elif self.feature_mode == 'both':
            loss = self.theano_sentence_loss(Sentence, Characters, WordLengths, GoldPredictions)
            input_list = [Sentence, Characters, WordLengths, GoldPredictions] + list(weight_list)
            
        grads = T.grad(loss, weight_list)

        cgraph = theano.function(inputs=input_list, outputs=grads, mode='FAST_RUN', allow_input_downcast=True)
        
        print("Done building graph")

        return cgraph
        
    '''
    Training and prediction:
    '''
    
    def train(self, sentences, labels, dev_sentences, dev_labels):
        self.optimizer.set_training_data(sentences, labels)
        self.optimizer.set_development_data(dev_sentences, dev_labels)

        self.optimizer.update()


    def pad_words(self, sentence_list):
        l = []
        for sentence in sentence_list:
            longest_word = max([len(x) for x in sentence])
            new_sentence = np.zeros((len(sentence), longest_word, len(sentence[0][0])))

            for i, word in enumerate(sentence):
                new_sentence[i, :len(word), :] = np.array(word)

            l.append(new_sentence)

        return l


    def predict(self, sentences):

        predict_function = self.build_predict_graph()
        
        if 'character' in sentences:       
            sentences['character'] = self.pad_words(sentences['character'])          
            word_lengths = [np.zeros(len(sentence)) for sentence in sentences['character']]

            for i,sentence in enumerate(sentences['character']):
                for j,word in enumerate(sentence):
                    word_lengths[i][j] = len(word)

                word_lengths[i] = word_lengths[i].astype(np.int32)
        
        predictions = []
        if not 'character' in sentences:
            for i,sentence in enumerate(sentences['sentence']):
                
                predictions.append(predict_function(sentence, *self.get_weight_list()))
                
        elif not 'sentence' in sentences:
            for chars, word_length in zip(sentences['character'], word_lengths):
                predictions.append(predict_function(chars, word_length, *self.get_weight_list()))

        else:
            for sentence, chars, word_length in zip(sentences['sentence'],
                                                    sentences['character'],
                                                    word_lengths):
                predictions.append(predict_function(sentence, chars, word_length, *self.get_weight_list()))

        return predictions

    '''
    Persistence:
    '''

    def save_graph(self, graph, filename):
        outfile = open(filename, 'wb')
        pickle.dump(graph, outfile)
        outfile.close()

    def load_graph(self, filename):
        infile = open(filename, 'rb')
        graph = pickle.load(infile)
        infile.close()

        return graph
        
        
    def save(self, filename):
        store_list = self.get_weight_list()
        
        outfile1 = open(filename, 'wb')
        pickle.dump(store_list, outfile1)
        outfile1.close()

        
    def load(self, filename):
        infile = open(filename, 'rb')
        store_list = pickle.load(infile)
        infile.close()

        self.update_weights(store_list)

        if self.optimizer is not None:
            self.optimizer.set_initial_weights(self.get_weight_list())

