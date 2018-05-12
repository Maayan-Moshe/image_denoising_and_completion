
import tensorflow as tf
from .height_weighter import HeightAverager, HeightDeviator
from .weight_averager import WeightAverager
from utils.loss_producer import LossTrainingProducer

PADDINGS = tf.constant([[0, 0,], [1, 1,], [1, 1]])

def prepare_graph(params = {'image shape': (155, 208), 'cost': 'l2_cost', 'num_of_layers': 2}):
    
    graph_dict = GraphCreator(params).create(params['num_of_layers'])
    return graph_dict
    
class GraphCreator:
    
    def __init__(self, params = {'image shape': (28, 28), 'cost': 'l2_cost'}):
        
        self.num_rows, self.num_cols = params['image shape']
        self.height_weighter_3x3 = HeightAverager(3)
        self.height_weighter_5x5 = HeightDeviator(5, self.num_rows, self.num_cols)
        self.weight_averager_9 = WeightAverager(9)
        self.weight_averager_25 = WeightAverager(25)
        self.loss_producer = LossTrainingProducer(params)
        
    def create(self, num_of_layers):
        
        input_layer = tf.placeholder(tf.float32, [None, self.num_rows, self.num_cols, 4], name = 'input')
        self.txy = input_layer[:, :, :, :3]
        
        predicted_z, self.weights_3x3 = self.__get_fundamental_layer(input_layer)
        predicted_z, self.weights_5x5 = self.__get_derivative_layer(predicted_z, self.weights_3x3, 0)
        
        for index in range(1,num_of_layers):
            predicted_z = self.__get_basic_layer(predicted_z, index)
        
        ans_dict = self.loss_producer.get_loss(predicted_z)
        ans_dict.update({'z_predicted': predicted_z, 'input': input_layer})
        return ans_dict
        
    def __get_basic_layer(self, pre_heights, index):
        with tf.name_scope('basic_layer_{}'.format(index)):
            z_3x3, w_3x3 = self.__get_averaging_layer(pre_heights, self.weights_5x5, index)
            self.weights_3x3 = self.weight_averager_9.average(self.weights_3x3, w_3x3)
            z_5x5, w_5x5 = self.__get_derivative_layer(z_3x3, self.weights_3x3, index)
            self.weights_5x5 = self.weight_averager_25.average(self.weights_5x5, w_5x5)
            tf.contrib.layers.summarize_tensors([z_3x3, w_3x3, z_5x5, w_5x5])
            return z_5x5
    
    def __get_fundamental_layer(self, input_layer):
        '''
        Fundamental layer comrised of: 2d colvolution with relu activation, 2D convolution to 
        produce weights, squashing the weights, weighting the heights.
        
        returns predicted heights.
        '''
        with tf.name_scope('fundamental_layer'):
            
            num_of_filters = 9*4*2
            conv_basic = tf.layers.conv2d(input_layer, filters = num_of_filters, kernel_size= 3, 
                                     padding="same", activation=tf.nn.relu, name = 'basic_conv',
                                     kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                     bias_initializer = tf.initializers.truncated_normal(mean = 1e-1, stddev = 2e-2))
            
            weights = tf.layers.conv2d(conv_basic, filters = 9, kernel_size = 1,
                                            padding="same", activation = None, 
                                            name = 'conv_{}_9'.format(num_of_filters),
                                            kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                            bias_initializer = tf.initializers.truncated_normal(mean = 1e-1, stddev = 2e-2))
            
            positive_weights = tf.nn.relu(weights, name = 'positive_weights')
        
            predicted_z = self.height_weighter_3x3.get_averaged_heights(input_layer[:,:,:,3], positive_weights)
            return predicted_z, weights
        
    def __get_derivative_layer(self, pre_heights, weights_3x3, index):
        '''
        This is a layer of spatial averaging of 5x5 allowing negative weights hence derivative.
        '''
        with tf.name_scope('derivative_layer_{}'.format(index)):
            
            stacked_input = self.__get_stacked_input(pre_heights, weights_3x3)

            conv_basic = tf.layers.conv2d(stacked_input, filters = 50, kernel_size= 5, reuse=False,
                                     padding="same", activation=tf.nn.relu, name = 'conv_10_50_{}'.format(index),
                                     kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                     bias_initializer = tf.initializers.truncated_normal(mean = 1e-1, stddev = 2e-2))
            
            weights = tf.layers.conv2d(conv_basic, filters = 25, kernel_size = 1,
                                            padding="same", activation = None, reuse=False,
                                            name = 'conv_50_25_{}'.format(index),
                                            kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                            bias_initializer = tf.initializers.truncated_normal(mean = 1e-1, stddev = 2e-2))
        
            predicted_z = self.height_weighter_5x5.get_averaged_heights(pre_heights, weights)
            tf.contrib.layers.summarize_tensors([predicted_z])
            return predicted_z, weights
        
    def __get_averaging_layer(self, pre_heights, weights_5x5, index):
        '''
        This is a layer of spatial averaging of 3x3 allowing only nonnegative weights hence averaging.
        '''
        with tf.name_scope('averaging_layer_{}'.format(index)):
            
            stacked_input = self.__get_stacked_input(pre_heights, weights_5x5)
            
            conv_basic = tf.layers.conv2d(stacked_input, filters = 52, kernel_size= 3, reuse=False,
                                     padding="same", activation=tf.nn.relu, name = 'conv_26_52_{}'.format(index),
                                     kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                     bias_initializer = tf.initializers.truncated_normal(mean = 1e-1, stddev = 2e-2))
            
            weights = tf.layers.conv2d(conv_basic, filters = 9, kernel_size = 1,
                                            padding="same", activation = None, reuse=False,
                                            name = 'conv_52_9_{}'.format(index),
                                            kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                            bias_initializer = tf.initializers.truncated_normal(mean = 1e-1, stddev = 2e-2))
            
            positive_weights = tf.nn.relu(weights, name = 'positive_weights')
        
            predicted_z = self.height_weighter_3x3.get_averaged_heights(pre_heights, positive_weights)
            return predicted_z, weights
            
    def __get_stacked_input(self, heights, weights):
        
        with tf.name_scope('stacking_input'):
            positive_weights = tf.nn.relu(weights, name = 'positive_weights')
            reshaped_z = tf.reshape(heights, shape = [-1, self.num_rows, self.num_cols, 1], name = 'reshaped_z')
            stacked_input = tf.concat([positive_weights, self.txy, reshaped_z], axis = 3, name = 'stacked_input')
            return stacked_input
                