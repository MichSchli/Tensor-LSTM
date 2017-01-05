import numpy as np
from theano import tensor as T
import theano
import pickle
import imp
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

superclass = imp.load_source('abstract_rnn', 'code/parsing/algorithms/abstract_rnn.py')
network_ops = imp.load_source('network_ops', 'code/parsing/algorithms/network_ops.py')
optimizers = imp.load_source('optimizers', 'code/parsing/algorithms/optimizers.py')

class FourwayLstm(superclass.RNN):

    '''
    Fields:
    '''

    hidden_dimension = 200
    input_dimension = 81
    srng = None
    use_cross_entropy_loss = True

    '''
    Initialization:
    '''

    def __init__(self, optimizer_config_path, loss="cross-entropy"):
        n_layers = 4

        self.input_lstm_layer = network_ops.fourdirectional_lstm_layer('input_layer_', self.input_dimension * 2 + 1, self.hidden_dimension)

        self.lstm_layers = [network_ops.fourdirectional_lstm_layer('layer_'+str(l),
                                                              self.hidden_dimension * 4,
                                                              self.hidden_dimension) for l in range(n_layers-1)]
        
        self.output_convolution = network_ops.linear_layer_on_tensor('output_layer', self.hidden_dimension * 4, 1)
        
        self.layers = [self.input_lstm_layer] + self.lstm_layers + [self.output_convolution]

        self.use_cross_entropy_loss = loss == "cross-entropy"

        super().__init__('sentence', optimizer_config_path)

        
    '''
    Theano functions:
    '''
        
    def __pairwise_features(self, V, Vs, sentence_length):
        thingy, _ = theano.scan(fn=lambda x, y: T.concatenate([y,T.zeros(1),x]),
                                sequences=Vs,
                                non_sequences=V)

        root_feature = T.concatenate((T.ones(1), T.zeros(self.input_dimension)))
        root_features = T.concatenate((V,root_feature))

        flat_version = thingy.flatten()
        with_root = T.concatenate((root_features, flat_version))
        in_shape = T.reshape(with_root, newshape=(sentence_length+1,self.input_dimension*2+1))
        return in_shape

    
   
    def theano_sentence_loss(self, Vs, gold):
        preds = self.theano_sentence_prediction(Vs)
        
        if self.use_cross_entropy_loss:
            losses = T.nnet.categorical_crossentropy(preds, gold)
        else:
            losses = T.pow(preds-gold,2)

        return T.sum(losses)


    def dropout(self, tensor, dropout_prob=0.5, training=True):
        if not training:
            return tensor

        if self.srng is None:
            self.srng = RandomStreams(seed=12345)

        keep_prob = 1.0 - dropout_prob
        mask = self.srng.binomial(size=tensor.shape, p=keep_prob, dtype='floatX')
        return tensor * mask / keep_prob
    
    def theano_sentence_prediction(self, Vs):
        pairwise_vs, _ = theano.scan(fn=self.__pairwise_features,
                                  outputs_info=None,
                                  sequences=Vs,
                                  non_sequences=[Vs, Vs.shape[0]])

        
        pairwise_vs = self.dropout(pairwise_vs, dropout_prob=0.2, training=self.input_lstm_layer.training)        
        
        full_matrix = self.input_lstm_layer.function(pairwise_vs)

        for layer in self.lstm_layers:
            full_matrix = self.dropout(full_matrix, dropout_prob=0.5, training=self.input_lstm_layer.training)            
            full_matrix = layer.function(full_matrix)

        full_matrix = self.dropout(full_matrix, dropout_prob=0.5, training=self.input_lstm_layer.training)
        
        final_matrix = self.output_convolution.function(full_matrix)[:,:,0]
        
        if self.use_cross_entropy_loss:
            final_matrix = T.nnet.softmax(final_matrix)

        return final_matrix

    

def fit(features, labels, dev_features, dev_labels, model_path=None, loss="cross-entropy"):
    optimizer_config_path = 'fourway_optimizer.config'    
    model = FourwayLstm(optimizer_config_path, loss=loss)

    model.save_path = model_path
    model.train(features, labels, dev_features, dev_labels)
        
def predict(features, model_path=None):
    model = FourwayLstm(None)
    model.load(model_path)

    predictions = model.predict(features)
    
    return predictions
