from theano import tensor as T
import theano
import numpy as np

class single_lstm(): 

    training=False
    
    def __init__(self, name, input_neurons, output_neurons):
        self.input_neurons=input_neurons
        self.output_neurons=output_neurons

        self.name = name

        #Initialize theano variables:
        self.W_forget_theano = T.fmatrix(self.name + '_forget_weight')
        self.W_input_theano = T.fmatrix(self.name + '_input_weight')
        self.W_candidate_theano = T.fmatrix(self.name + '_candidate_weight')
        self.W_output_theano = T.fmatrix(self.name + '_output_weight')

        #Initialize python variables:

        high_init = np.sqrt(6)/np.sqrt(self.input_neurons + 2*self.output_neurons)
        low_init = -high_init
        
        s = (self.output_neurons, self.input_neurons + self.output_neurons + 1)
        self.W_forget = np.random.uniform(low=low_init, high=high_init, size=s).astype(np.float32)
        self.W_input = np.random.uniform(low=low_init, high=high_init, size=s).astype(np.float32)
        self.W_candidate = np.random.uniform(low=low_init, high=high_init, size=s).astype(np.float32)
        self.W_output = np.random.uniform(low=low_init, high=high_init, size=s).astype(np.float32)

        #Initialize forget bias to one:
        self.W_forget[-1] = np.ones_like(self.W_forget[-1], dtype=np.float32)

    def set_training(self, training):
        self.training=training

    def update_weights(self, update_list):
        self.W_forget = update_list[0]
        self.W_input = update_list[1]
        self.W_candidate = update_list[2]
        self.W_output = update_list[3]

    def weight_count(self):
        return 4
        
    def get_theano_weights(self):
        return self.W_forget_theano, self.W_input_theano, self.W_candidate_theano, self.W_output_theano

    def get_python_weights(self):
        return self.W_forget, self.W_input, self.W_candidate, self.W_output
    
    
    def function(self, x, h_prev, c_prev):
        input_vector = T.concatenate((x, h_prev, T.ones(1)))
        
        forget_gate = T.nnet.sigmoid(T.dot(self.W_forget_theano, input_vector))
        input_gate = T.nnet.sigmoid(T.dot(self.W_input_theano, input_vector))
        candidate_vector = T.tanh(T.dot(self.W_candidate_theano, input_vector))
        cell_state = forget_gate*c_prev + input_gate * candidate_vector

        output = T.nnet.sigmoid(T.dot(self.W_output_theano, input_vector))
        h = output * T.tanh(cell_state)
        return h, cell_state

class single_lstm_on_tensor(single_lstm):

    def function(self, xs, h_prevs, c_prevs):
        biases = T.shape_padright(T.ones_like(xs[:,0]))
        input_vector = T.concatenate((xs, h_prevs, biases), axis=1)

        forget_gate = T.nnet.sigmoid(T.tensordot(input_vector, self.W_forget_theano, axes=[[1],[1]]))
        input_gate = T.nnet.sigmoid(T.tensordot(input_vector, self.W_input_theano, axes=[[1],[1]]))
        candidate_vector = T.tanh(T.tensordot(input_vector, self.W_candidate_theano, axes=[[1],[1]]))
        cell_state = forget_gate*c_prevs + input_gate * candidate_vector

        output = T.nnet.sigmoid(T.tensordot(input_vector, self.W_output_theano, axes=[[1],[1]]))
        h = output * T.tanh(cell_state)
        return h, cell_state
    
class lstm_layer():

    training=False
    
    def __init__(self, name, input_neurons, output_neurons, direction=True, get_cell_values=False):
        self.input_neurons=input_neurons
        self.output_neurons=output_neurons
        self.direction=direction

        self.name = name
        self.neuron=single_lstm_on_tensor(name, input_neurons, output_neurons)

        self.get_cell_values=get_cell_values
    
    def set_training(self, training):
        self.training=training
        self.neuron.set_training(training)
        
    def update_weights(self, update_list):
        self.neuron.update_weights(update_list)

    def weight_count(self):
        return self.neuron.weight_count()
    
    def get_theano_weights(self):
        return self.neuron.get_theano_weights()

    def get_python_weights(self):
        return self.neuron.get_python_weights()
        
    def function(self, Vs):
        h0 = T.zeros((Vs.shape[1], self.output_neurons))
        c0 = T.zeros((Vs.shape[1], self.output_neurons))

        lstm_preds, _ = theano.scan(fn=self.neuron.function,
                        outputs_info=[h0,c0],
                        sequences=Vs,
                        non_sequences=None,
                        go_backwards=not self.direction)

        if self.direction:
            if not self.get_cell_values:
                return lstm_preds[0]
            else:
                return lstm_preds
        else:
            if not self.get_cell_values:
                return lstm_preds[0][::-1]
            else:
                return [lstm_preds[0][::-1], lstm_preds[1][::-1]]

            
class bidirectional_rnn_lstm():

    training=False
    
    def __init__(self, name, input_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons

        self.name = name
        self.forward = lstm_layer(name + '_forward', input_neurons, output_neurons, True)
        self.backward = lstm_layer(name + '_backward', input_neurons, output_neurons, False)
    
    def set_training(self, training):
        self.training=training
        self.forward.set_training(training)
        self.backward.set_training(training)
        
    def update_weights(self, update_list):
        self.forward.update_weights(update_list[:self.forward.weight_count()])
        self.backward.update_weights(update_list[self.forward.weight_count():])

    def weight_count(self):
        return self.forward.weight_count() + self.backward.weight_count()
        
    def get_theano_weights(self):
        return self.forward.get_theano_weights() + self.backward.get_theano_weights()

    def get_python_weights(self):
        return self.forward.get_python_weights() + self.backward.get_python_weights()
        
    
    def function(self, Vs):
        
        forwards_h = self.forward.function(Vs)[-1]
        backwards_h = self.backward.function(Vs)[-1]

        return T.concatenate((forwards_h, backwards_h))


class bidirectional_lstm_layer():

    training=False
    
    def __init__(self, name, input_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons

        self.name = name
        self.forward = lstm_layer(name + '_forward', input_neurons, output_neurons, True)
        self.backward = lstm_layer(name + '_backward', input_neurons, output_neurons, False)
    
    def set_training(self, training):
        self.training=training
        self.forward.set_training(training)
        self.backward.set_training(training)
        
    def update_weights(self, update_list):
        self.forward.update_weights(update_list[:self.forward.weight_count()])
        self.backward.update_weights(update_list[self.forward.weight_count():])

    def weight_count(self):
        return self.forward.weight_count() + self.backward.weight_count()
        
    def get_theano_weights(self):
        return self.forward.get_theano_weights() + self.backward.get_theano_weights()

    def get_python_weights(self):
        return self.forward.get_python_weights() + self.backward.get_python_weights()
        
    
    def function(self, Vs):
        
        forwards_h = self.forward.function(Vs)
        backwards_h = self.backward.function(Vs)

        return T.concatenate((forwards_h, backwards_h), axis=2)

    
class fourdirectional_lstm_layer():

    training=False
    
    def __init__(self, name, input_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons

        self.name = name
        self.sideward_layer = bidirectional_lstm_layer(name + '_sideward', input_neurons, output_neurons)
        self.downward_layer = bidirectional_lstm_layer(name + '_downward', input_neurons, output_neurons)

    
    def set_training(self, training):
        self.training=training
        self.sideward_layer.set_training(training)
        self.downward_layer.set_training(training)
        

    def update_weights(self, update_list):
        self.sideward_layer.update_weights(update_list[:self.sideward_layer.weight_count()])
        self.downward_layer.update_weights(update_list[self.downward_layer.weight_count():])

    def weight_count(self):
        return self.sideward_layer.weight_count() + self.downward_layer.weight_count()
        
    def get_theano_weights(self):
        return self.sideward_layer.get_theano_weights() + self.downward_layer.get_theano_weights()

    def get_python_weights(self):
        return self.sideward_layer.get_python_weights() + self.downward_layer.get_python_weights()
        
    
    def function(self, VM):
        lstm_sidewards = self.sideward_layer.function(VM)
        
        transpose_vm = VM.transpose(1,0,2)
        lstm_downwards = self.downward_layer.function(transpose_vm)
        
        return T.concatenate((lstm_sidewards,lstm_downwards.transpose(1,0,2)), axis=2)


    
class linear_layer_on_tensor():

    training=False
    
    def __init__(self, name, input_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.name = name

        high_init = np.sqrt(6)/np.sqrt(input_neurons + output_neurons)
        low_init = -np.sqrt(6)/np.sqrt(input_neurons + output_neurons)
    
        self.weight_matrix_theano = T.fmatrix(name + '_weight')
        self.weight_matrix = np.random.uniform(low=low_init, high=high_init, size=(self.output_neurons, self.input_neurons+1)).astype(np.float32)

    def set_training(self, training):
        self.training=training
    
    def update_weights(self, update_list):
        self.weight_matrix = update_list[0]
        
    def weight_count(self):
        return 1
            
    def get_theano_weights(self):
        return (self.weight_matrix_theano,)

    def get_python_weights(self):
        return (self.weight_matrix,)
        
    def function(self, input_tensor):
        biases = T.shape_padright(T.ones_like(input_tensor[:,:,0]))
        input_with_bias = T.concatenate((input_tensor, biases), axis=2)
        return T.tensordot(input_with_bias, self.weight_matrix_theano, axes=[[2],[1]])

