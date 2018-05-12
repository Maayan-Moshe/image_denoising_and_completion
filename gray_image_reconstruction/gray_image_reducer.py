
import tensorflow as tf
from utils.utils import get_normalized_weights, stack_input, reshape_image

class HeightMapReducerBase:
    
    def __init__(self, num_rows, num_cols):
        
        self.num_rows = num_rows
        self.num_cols = num_cols
    
    def get_reduced(self, previous_data):
        
        raise NotImplementedError
        
    def _get_reduced_data(self, previous_data):
        
        stacked_input = stack_input(previous_data)
        weights = get_weights(stacked_input)
        normed_weights = get_normalized_weights(weights)
        reduced_stacked_z = get_padded_reduced_z(previous_data['image'], self.num_rows, self.num_cols)
        av_image, weighted_height_map = weight_heights(reduced_stacked_z, normed_weights)
        return av_image, weights, weighted_height_map, reduced_stacked_z


class HeightMapReducer(HeightMapReducerBase):
    
    def __init__(self, num_rows, num_cols, params):
        
        HeightMapReducerBase.__init__(self, num_rows, num_cols)
        
    def get_reduced(self, previous_data):
        
        av_image, weights, _, _ = self._get_reduced_data(previous_data)
        
        return {'image': av_image, 'weights': weights}
            
class HeightMapReducerFiller(HeightMapReducerBase):
    
    def __init__(self, num_rows, num_cols, params = {'regularization': {'kernel': 0.5, 'bias': 0.5}}):
        
        HeightMapReducerBase.__init__(self, num_rows, num_cols)
        self.reg = params['regularization']
        
    def get_reduced(self, previous_data):
        
        av_image, av_weights, weighted_image, reduced_stacked_z = self._get_reduced_data(previous_data)
        addition_image, add_weights = self.__addition_height_map(weighted_image, reduced_stacked_z, av_image)
        weights = tf.concat([av_weights, add_weights], axis = 3)
        reduced_image = av_image + addition_image
        return {'image': reduced_image, 'weights': weights, 'addition_image': addition_image}
        
    def __addition_height_map(self, weighted_height_map, reduced_stacked_z, av_height_map):
        with tf.name_scope('additional_height_map'):
            stacked_input = tf.concat([weighted_height_map, reduced_stacked_z, av_height_map], axis = 3)
            
            first_layer = tf.contrib.layers.conv2d(inputs = stacked_input, stride = 1,
                                    num_outputs = 32, kernel_size = 3, activation_fn = tf.nn.relu,
                                    weights_regularizer = tf.contrib.layers.l2_regularizer(self.reg['kernel']),
                                    biases_regularizer = tf.contrib.layers.l2_regularizer(self.reg['bias']), 
                                    biases_initializer=tf.initializers.truncated_normal(mean = 2e-3, stddev = 4e-4))
                                    
            second_layer = tf.contrib.layers.conv2d(inputs = first_layer, stride = 1,
                                        num_outputs = 8, kernel_size = 3, activation_fn = tf.nn.relu,
                                        weights_regularizer = tf.contrib.layers.l2_regularizer(self.reg['kernel']),
                                        biases_regularizer = tf.contrib.layers.l2_regularizer(self.reg['bias']), 
                                        biases_initializer=tf.initializers.truncated_normal(mean = 2e-3, stddev = 4e-4))
                                        
            addition_image = tf.contrib.layers.conv2d(inputs = second_layer, stride = 1,
                                        num_outputs = 1, kernel_size = 3, activation_fn = None,
                                        weights_regularizer = tf.contrib.layers.l2_regularizer(self.reg['kernel']),
                                        biases_regularizer = tf.contrib.layers.l2_regularizer(self.reg['bias']), 
                                        biases_initializer=tf.initializers.truncated_normal(mean = 2e-3, stddev = 4e-4))
                                    
            return addition_image, second_layer
                                        
def get_padded_reduced_z(input_z, num_rows, num_cols):
    '''
    Padding z so we could convientley choose slices.
    '''
    with tf.name_scope('padding_reducing_heights'):
        pad_rows = num_rows%2
        pad_cols = num_cols%2
        padding = tf.constant([[0, 0,], [0, pad_rows,], [0, pad_cols], [0, 0]])
        padded_z = tf.pad(reshape_image(input_z), padding, "SYMMETRIC")
        reduced_stacked_z = tf.space_to_depth(padded_z, 2, name = 'reduced_stacked_z')
        return reduced_stacked_z

def get_weights(stacked_input):
    with tf.name_scope('calculating_weights'):
        first_layer = tf.contrib.layers.conv2d(inputs = stacked_input, stride = 2,
                                num_outputs = 64, kernel_size = 4, activation_fn = tf.nn.relu,
                                biases_initializer=tf.initializers.truncated_normal(mean = 2e-1, stddev = 4e-2))

        weights = tf.contrib.layers.conv2d(inputs = first_layer, stride = 1,
                            num_outputs = 4, kernel_size = 3, activation_fn = tf.nn.relu,
                            biases_initializer=tf.initializers.truncated_normal(mean = 2e-1, stddev = 4e-2))

        return weights

def weight_heights(reduced_stacked_z, normed_weights):
    with tf.name_scope('weighting_heights'):
        weighted_height_map = reduced_stacked_z*normed_weights
        reduced_image = tf.reduce_sum(weighted_height_map, axis = 3)
        reduced_image = tf.expand_dims(reduced_image, axis = 3)
        return reduced_image, weighted_height_map
        
if __name__ == '__main__':
    import sys
    sys.modules[__name__]