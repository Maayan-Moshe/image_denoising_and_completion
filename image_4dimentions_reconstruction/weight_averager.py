
import tensorflow as tf

class WeightAverager:
    
    def __init__(self, num_channels):
        with tf.name_scope('averaging_weights'):
            self.num_channels = num_channels
            self.bias = tf.get_variable("bias_{}".format(num_channels), [self.num_channels], 
                                initializer=tf.initializers.truncated_normal(stddev = 1e-1, mean = 1e-1))
        
    def average(self, pre_weights, current_weights):
        '''
        Returning weighted average of the weights from two instances.
        Parameters are reused from layer to layer.
        '''
        with tf.name_scope('averaging_weights'):
            beta_pre = tf.contrib.layers.convolution2d(inputs = pre_weights, 
                                        num_outputs = self.num_channels, kernel_size = 1,
                                        activation_fn = None, 
                                        reuse = False, biases_initializer = None,
                                        weights_initializer = tf.initializers.truncated_normal(stddev = 1e-1))
            
            beta_post = tf.contrib.layers.convolution2d(inputs = current_weights, 
                                        num_outputs = self.num_channels, kernel_size = 1,
                                        activation_fn = None, 
                                        reuse = False, biases_initializer = None,
                                        weights_initializer = tf.initializers.truncated_normal(stddev = 1e-1))
            
            pre_w = beta_pre + beta_post + self.bias
            average_weights = pre_w*pre_weights + (1-pre_w)*current_weights
            return average_weights
        