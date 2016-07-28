import sys
import numpy as np

class Optimizer():

    training_sentences = {}
    development_sentences = {}
    
    gradient_clipping_factor = None
    use_gradient_noise = False
    gradient_noise_eta = None
    gradient_noise_gamma = None

    def initialize(self):
        pass

    def set_loss_function(self, function):
        self.loss_function = function

    def set_gradient_function(self, function):
        self.gradient_function = function

    def set_initial_weights(self, weights):
        self.weights = weights
        
    def set_update_function(self, function):
        self.update_function = function 

    def do_update(self):
        self.update_function(self.weights)
        
    def set_training_data(self, sentences, labels):
        self.training_labels = labels
            
        if 'character' in sentences:       
            self.training_sentences['character'] = self.pad_words(sentences['character'])
          
            self.training_word_lengths = [np.zeros(len(sentence)) for sentence in sentences['character']]

            for i,sentence in enumerate(sentences['character']):
                for j,word in enumerate(sentence):
                    self.training_word_lengths[i][j] = len(word)

                self.training_word_lengths[i] = self.training_word_lengths[i].astype(np.int32)

        if 'sentence' in sentences:       
            self.training_sentences['sentence'] = sentences['sentence']



    def set_development_data(self, sentences, labels):
        self.development_labels = labels
            
        if 'character' in sentences:       
            self.development_sentences['character'] = self.pad_words(sentences['character'])
          
            self.development_word_lengths = [np.zeros(len(sentence)) for sentence in sentences['character']]

            for i,sentence in enumerate(sentences['character']):
                for j,word in enumerate(sentence):
                    self.development_word_lengths[i][j] = len(word)

                self.development_word_lengths[i] = self.development_word_lengths[i].astype(np.int32)

        if 'sentence' in sentences:       
            self.development_sentences['sentence'] = sentences['sentence']


        
    def development_loss(self):
        loss = 0

        if not 'character' in self.development_sentences:
            for sentence, label in zip(self.development_sentences['sentence'],
                                       self.development_labels):
                loss += self.loss_function(sentence, label, *self.weights)
                

        elif not 'sentence' in self.development_sentences:
            for chars, word_lengths, label in zip(self.development_sentences['character'],
                                                  self.development_word_lengths,
                                                  self.development_labels):
                loss += self.loss_function(chars, word_lengths, label, *self.weights)

        else:
            for sentence, chars, word_lengths, label in zip(self.development_sentences['sentence'],
                                                            self.development_sentences['character'],
                                                            self.development_word_lengths,
                                                            self.development_labels):
                loss += self.loss_function(sentence, chars, word_lengths, label, *self.weights)
        
        return loss
    
    def process_gradients(self, gradients):
        if self.use_gradient_noise:
            std_dev = np.sqrt(self.gradient_noise_eta/(self.current_iteration**self.gradient_noise_gamma))

            gradients = [g + np.random.normal(0, std_dev, size=g.shape) for g in gradients]
        
        if self.gradient_clipping_factor is not None:
            gradient_l2_norm = np.sqrt(sum([np.square(g).sum() for g in gradients]))
            if gradient_l2_norm > self.gradient_clipping_factor:
                gradients = [g*float(self.gradient_clipping_factor)/gradient_l2_norm for g in gradients]

        return gradients


    '''
    Padding:
    '''

    def pad_words(self, sentence_list):
        l = []
        for sentence in sentence_list:
            longest_word = max([len(x) for x in sentence])
            new_sentence = np.zeros((len(sentence), longest_word, len(sentence[0][0])))

            for i, word in enumerate(sentence):
                new_sentence[i, :len(word), :] = np.array(word)

            l.append(new_sentence)

        return l
            
    
class MinibatchOptimizer(Optimizer):

    batch_size = None
    max_iterations = None
    
    error_margin = 10**(-8)
    normalize_batches = True

    def initialize(self):
        if self.batch_size is None or self.max_iterations is None:
            raise Exception("Optimizer not fully specified in config file!")

        super().initialize()

    def batch_gradients_s(self, data_batch, label_batch):
        aggregate = None
        for sentence, gold in zip(data_batch, label_batch):
            if aggregate is None:
                aggregate = self.gradient_function(sentence, gold, *self.weights)
            else:
                new_g = self.gradient_function(sentence, gold, *self.weights)
                aggregate = [aggregate[i] + new_g[i] for i in range(len(new_g))]

        return aggregate

    def batch_gradients_c(self, data_batch, word_length_batch, label_batch):
        aggregate = None
        for sentence, word_lengths, gold in zip(data_batch, word_length_batch, label_batch):
            if aggregate is None:
                aggregate = self.gradient_function(sentence, word_lengths, gold, *self.weights)
            else:
                new_g = self.gradient_function(sentence, word_lengths, gold, *self.weights)
                aggregate = [aggregate[i] + new_g[i] for i in range(len(new_g))]

        return aggregate
        
    def batch_gradients_b(self, data_batch, char_batch, word_length_batch, label_batch):
        aggregate = None
        for sentence, chars, word_lengths, gold in zip(data_batch, char_batch, word_length_batch, label_batch):
            if aggregate is None:
                aggregate = self.gradient_function(sentence, chars, word_lengths, gold, *self.weights)
            else:
                new_g = self.gradient_function(sentence, chars, word_lengths, gold, *self.weights)
                aggregate = [aggregate[i] + new_g[i] for i in range(len(new_g))]

        return aggregate
        
    def process_gradients(self, gradients):
        if self.normalize_batches:
            gradients = [g/self.batch_size for g in gradients]
        
        gradients = super().process_gradients(gradients)
        return gradients
        
    def chunk(self, l):
        return np.array([l[i:i+self.batch_size] for i in range(0, len(l), self.batch_size)])
        
    def update(self):
        self.current_iteration = 1
        
        if 'sentence' in self.training_sentences:
            sentence_chunks = self.chunk(self.training_sentences['sentence'])

        if 'character' in self.training_sentences:
            character_chunks = self.chunk(self.training_sentences['character'])
            word_length_chunks = self.chunk(self.training_word_lengths)

        label_chunks = self.chunk(self.training_labels)

        current_loss = self.development_loss()
        prev_loss = current_loss +1

        self.updates = [np.zeros_like(weight) for weight in self.weights]
        
        while(self.current_iteration < self.max_iterations and prev_loss > current_loss):
            prev_loss = current_loss
            print("Running optimizer at iteration "+str(self.current_iteration)+". Current loss: "+str(prev_loss))
            self.current_iteration += 1

            if not 'sentence' in self.training_sentences:
                for data_batch, word_length_batch, label_batch in zip(character_chunks,
                                                                      word_length_chunks,
                                                                      label_chunks):
                    self.batch_update_c(data_batch, word_length_batch, label_batch)

                    for i, update in enumerate(self.updates):
                        self.weights[i] += self.updates[i]

            if not 'character' in self.training_sentences:
                for data_batch, label_batch in zip(sentence_chunks, label_chunks):
                    self.batch_update_s(data_batch, label_batch)

                    for i, update in enumerate(self.updates):
                        self.weights[i] += self.updates[i]

            else:
                for data_batch, char_batch, word_length_batch, label_batch in zip(sentence_chunks,
                                                                                  character_chunks,
                                                                                  word_length_chunks,
                                                                                  label_chunks):
                    self.batch_update_b(data_batch, char_batch, word_length_batch, label_batch)

                    for i, update in enumerate(self.updates):
                        self.weights[i] += self.updates[i]


            print('')
            #self.do_update()

            current_loss = self.development_loss()
            if prev_loss > current_loss:
                self.do_update()

        print("Stopping.")

class StochasticGradientDescent(MinibatchOptimizer):

    def initialize(self):
        if self.learning_rate is None or self.momentum is None:
            raise Exception("Optimizer not fully specified in config file!")

        super().initialize()

    def batch_update_s(self, data_batch, label_batch):
        gradients = self.batch_gradients_s(data_batch, label_batch)
        self.__update_from_gradients(gradients)
    
    def batch_update_c(self, data_batch, word_length_batch, label_batch):
        gradients = self.batch_gradients_c(data_batch, word_length_batch, label_batch)
        self.__update_from_gradients(gradients)

    def batch_update_b(self, data_batch, char_batch, word_length_batch, label_batch):
        gradients = self.batch_gradients_b(data_batch, char_batch, word_length_batch, label_batch)
        self.__update_from_gradients(gradients)

        
    def __update_from_gradients(self, gradients):
        gradients = self.process_gradients(gradients)
        
        for i, gradient in enumerate(gradients):
            self.updates[i] = -self.learning_rate*gradient + self.momentum * self.updates[i]

        if self.verbose:
            print('.', end='', flush=True)


class AdaDelta(MinibatchOptimizer):

    decay_rate = None
    
    epsillon = 10**(-6)
    
    def initialize(self):
        if self.decay_rate is None:
            raise Exception("Optimizer not fully specified in config file!")

        super().initialize()
        
        self.running_average = [np.zeros_like(weight) for weight in self.weights]


    def batch_update_s(self, data_batch, label_batch):
        gradients = self.batch_gradients_s(data_batch, label_batch)
        self.__update_from_gradients(gradients)
    
    def batch_update_c(self, data_batch, word_length_batch, label_batch):
        gradients = self.batch_gradients_c(data_batch, word_length_batch, label_batch)
        self.__update_from_gradients(gradients)

    def batch_update_b(self, data_batch, char_batch, word_length_batch, label_batch):
        gradients = self.batch_gradients_b(data_batch, char_batch, word_length_batch, label_batch)
        self.__update_from_gradients(gradients)

        
    def __update_from_gradients(self, gradients):
        for i, gradient in enumerate(gradients):
            square_gradient = np.square(gradient)
            self.running_average[i] = self.decay_rate * self.running_average[i] + (1 - self.decay_rate) * square_gradient

            rmsx = np.sqrt(np.square(self.updates[i]) + self.epsillon)
            rmsgrad = np.sqrt(self.running_average[i] + self.epsillon)
            
            self.updates[i] = -rmsx / rmsgrad * gradient

        if self.verbose:
            print('.', end='', flush=True)

        

class RMSProp(MinibatchOptimizer):

    decay_rate = None
    learning_rate = None
    
    epsillon = 10**(-6)

    def initialize(self):
        if self.decay_rate is None or self.learning_rate is None:
            raise Exception("Optimizer not fully specified in config file!")

        super().initialize()
        
        self.running_average = [np.zeros_like(weight) for weight in self.weights]


    def batch_update_s(self, data_batch, label_batch):
        gradients = self.batch_gradients_s(data_batch, label_batch)
        self.__update_from_gradients(gradients)
    
    def batch_update_c(self, data_batch, word_length_batch, label_batch):
        gradients = self.batch_gradients_c(data_batch, word_length_batch, label_batch)
        self.__update_from_gradients(gradients)

    def batch_update_b(self, data_batch, char_batch, word_length_batch, label_batch):
        gradients = self.batch_gradients_b(data_batch, char_batch, word_length_batch, label_batch)
        self.__update_from_gradients(gradients)

        
    def __update_from_gradients(self, gradients):
        for i, gradient in enumerate(gradients):
            square_gradient = np.square(gradient)
            self.running_average[i] = self.decay_rate * self.running_average[i] + (1 - self.decay_rate) * square_gradient

            rmsgrad = np.sqrt(self.running_average[i] + self.epsillon)
            
            self.updates[i] = -self.learning_rate / rmsgrad * gradient

        if self.verbose:
            print('.', end='', flush=True)

        
def from_config(config):
    if 'algorithm' not in config:
        raise Exception('Optimization algorithm not specified!')

    if config['algorithm'] == 'SGD':
        optimizer = StochasticGradientDescent()
    elif config['algorithm'] == 'RMSProp':
        optimizer = RMSProp()
    elif config['algorithm'] == 'AdaDelta':
        optimizer = AdaDelta()
    else:
        raise Exception('Optimization algorithm \"'+config['algorithm']+'\" unknown!')

    if 'verbose' in config:
        optimizer.verbose = bool(config['verbose'])

    if 'max iterations' in config:
        optimizer.max_iterations = int(config['max iterations'])
        
    if 'batch size' in config:
        optimizer.batch_size = int(config['batch size'])

    if 'learning rate' in config:
        optimizer.learning_rate = float(config['learning rate'])

    if 'decay rate' in config:
        optimizer.decay_rate = float(config['decay rate'])
        
    if 'gradient clip' in config:
        optimizer.gradient_clipping_factor = float(config['gradient clip'])
    
    if 'gradient noise' in config:
        if 'noise eta' not in config or 'noise gamma' not in config:
            raise Exception('Gradient noise parameters not specified')
        optimizer.use_gradient_noise = bool(config['gradient noise'])
        optimizer.gradient_noise_eta = float(config['noise eta'])
        optimizer.gradient_noise_gamma = float(config['noise gamma'])

    return optimizer
