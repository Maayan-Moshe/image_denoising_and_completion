
from utils.utils import get_one_sum_kernel
import tensorflow as tf

def expand_data(reduced_average_data, pre_reduced_dat, pre_shape):
    with tf.name_scope('expanding_data'):
        expander = DataExpander(pre_shape)
        expanded_data = {key: expander.get_expand_gray_image(value, kernel_size = 3) \
                for key, value in reduced_average_data.items() if 'image' in key}
        expanded_data.update({'pre_' + key: expander.get_expand_gray_image(value, kernel_size = 3) \
                for key, value in pre_reduced_dat.items() if 'image' in key})
        expanded_data.update({'weights': get_weights(reduced_average_data['weights'], pre_reduced_dat['weights'], expander)})
        return expanded_data
        
def get_weights(average_weights, pre_weights, expander):
    with tf.name_scope('expanding_weights'):
        expanded_post_weights = expander.get_stacked_exapnded_weights(average_weights)
        expanded_pre_weights = expander.get_expand_weights(pre_weights)
        expanded_weights = tf.concat([expanded_pre_weights, expanded_post_weights], axis = 3, name = 'expanded_weights')
        return expanded_weights
       
class DataExpander:
    
    def __init__(self, pre_shape):
        
        self.ps = [-1, pre_shape[1].value, pre_shape[2].value, 1]
        
    def get_expand_gray_image(self, input_layer, kernel_size = 3):
        with tf.name_scope('expanding_image'):
            with tf.variable_scope('expanding_image', reuse = tf.AUTO_REUSE):
                kernel = get_one_sum_kernel(kernel_size, num_inputs = 1, num_outputs = 4)
                conv2 = tf.nn.conv2d(input_layer, kernel, strides = [1, 1, 1, 1], padding = 'SAME')
                                                        
                increased_paded = tf.depth_to_space(conv2, 2, name = 'increased')
        
                increased = tf.slice(increased_paded, [0, 0, 0, 0], self.ps)
           
                return increased
                
    def get_stacked_exapnded_weights(self, reduced_weights):
        with tf.name_scope('expanding_weights'):
            stacked_input = tf.concat([reduced_weights, reduced_weights, reduced_weights, reduced_weights], axis = 3, name = 'stacked_input')
            expanded_weights = self.get_expand_weights(stacked_input) 
            
            return expanded_weights
        
    def get_expand_weights(self, weights):
        with tf.name_scope('expanding_weights'):
            
            expanded_weights_padded = tf.depth_to_space(weights, 2, name = 'increased_weights')#TODO what happens with depth 8
            out_num_chan = expanded_weights_padded.get_shape()[3].value
            expanded_weights = tf.slice(expanded_weights_padded, [0,0,0,0], [-1, self.ps[1], self.ps[2], out_num_chan])
       
            return expanded_weights