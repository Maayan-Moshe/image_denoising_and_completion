
import tensorflow as tf
from .height_weighter import HeightDeviator, HeightAverager
from .weight_averager import WeightAverager
from utils.loss_producer import LossTrainingProducer

def prepare_graph(params = {'image shape': (155, 208), 'cost': 'l2_cost', 'num_of_layers': 2}):
    
    graph_dict = OnlyDerivativeGraphCreator(params).create(params['num_of_layers'])
    return graph_dict

class OnlyDerivativeGraphCreator:
    
    def __init__(self, params = {'image shape': (28, 28), 'cost': 'l2_cost'}):
        
        self.num_rows, self.num_cols = params['image shape']
        self.height_deviator_5x5 = HeightDeviator(5, self.num_rows, self.num_cols)
        self.height_averager_5x5 = HeightAverager(5)
        self.weight_averager_72 = WeightAverager(72)
        self.loss_producer = LossTrainingProducer(params)
        
    def create(self, num_of_basic_layers):
        
        input_layer = tf.placeholder(tf.float32, [None, self.num_rows, self.num_cols, 4], name = 'input')
        self.txy = input_layer[:, :, :, :3]
        self.__set_initial_weights(input_layer)
        
        predicted_z = input_layer[:, :, :, 3]
        for index in range(num_of_basic_layers):
            predicted_z = self.__advance_basic_layer(predicted_z, index)

        ans_dict = {'z_predicted': predicted_z, 'input': input_layer}
        ans_dict.update(self.loss_producer.get_loss(ans_dict))
        return ans_dict
        
    def __set_initial_weights(self, input_layer):
        '''
        We initilize the besic weights used in each layer
        '''
        with tf.name_scope('initial_weights'):
            self.weights_state = tf.contrib.layers.convolution2d(inputs = input_layer,
                                num_outputs = 72, kernel_size = 3, activation_fn=None,
                                biases_initializer=tf.initializers.truncated_normal(mean = 2e-1, stddev = 4e-2))
            
    def __advance_basic_layer(self, pre_z, index):
        
        with tf.name_scope('basic_layer_{}'.format(index)):
                        
            averaged_z, first_weights = self.__get_averaged_z(pre_z)
            
            second_weights = self.__set_weights_get_second_weights(averaged_z, first_weights)
            
            post_z = self.__get_derivative_z(averaged_z, second_weights)
            return post_z
        
    def __get_averaged_z(self, pre_z):
        
        with tf.name_scope('averaging_heights'):
            reshaped_z = tf.reshape(pre_z, shape = [-1, self.num_rows, self.num_cols, 1], name = 'reshaped_z')
            stacked_input = tf.concat([self.txy, reshaped_z, tf.nn.relu(self.weights_state)], axis = 3, name = 'stacked_input')
            first_weights = tf.contrib.layers.convolution2d(inputs = stacked_input,
                                num_outputs = 90, kernel_size = 3, activation_fn=tf.nn.relu,
                                biases_initializer=tf.initializers.truncated_normal(mean = 2e-1, stddev = 4e-2))
            
            average_weights = tf.contrib.layers.convolution2d(inputs = first_weights,
                                num_outputs = 25, kernel_size = 3, activation_fn=tf.nn.relu,
                                biases_initializer=tf.initializers.truncated_normal(mean = 2e-1, stddev = 4e-2))
            
            post_z = self.height_averager_5x5.get_averaged_heights(pre_z, average_weights)
            tf.contrib.layers.summarize_tensors([post_z, average_weights])
            return post_z, first_weights
        
    def __set_weights_get_second_weights(self, averaged_z, first_weights):
        
        with tf.name_scope('setting_weights'):
            reshaped_z = tf.reshape(averaged_z, shape = [-1, self.num_rows, self.num_cols, 1], name = 'reshaped_z')
            stacked_input = tf.concat([self.txy, reshaped_z, first_weights], axis = 3, name = 'stacked_input')
            second_weights = tf.contrib.layers.convolution2d(inputs = stacked_input,
                                num_outputs = 72, kernel_size = 3, activation_fn=None,
                                biases_initializer=tf.initializers.truncated_normal(mean = 2e-1, stddev = 4e-2))
            
            self.weights_state = self.weight_averager_72.average(self.weights_state, second_weights)
            return second_weights
        
    def __get_derivative_z(self, pre_z, second_weights):
        
        with tf.name_scope('derivative_heights'):
            av_inputs = tf.nn.relu(second_weights)
            average_weights = tf.contrib.layers.convolution2d(inputs = av_inputs, 
                                weights_regularizer = tf.contrib.layers.l2_regularizer(scale = 0.1),
                                biases_regularizer = tf.contrib.layers.l2_regularizer(scale = 0.1),
                                num_outputs = 25, kernel_size = 3, activation_fn=None,
                                biases_initializer=tf.initializers.truncated_normal(mean = 0, stddev = 4e-2))
            
            post_z = self.height_deviator_5x5.get_averaged_heights(pre_z, average_weights)
            tf.contrib.layers.summarize_tensors([post_z, average_weights])
            return post_z
            
        