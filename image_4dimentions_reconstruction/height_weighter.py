
import tensorflow as tf
import math
from utils.utils import get_norm_of_weights

class HeightWeighterBase:
    
    def __init__(self, win_size, min_height, max_height):
        '''
        Window size should be 1,3,5,7,....
        '''
        self.win_size = win_size
        self.min_height = min_height
        self.max_height = max_height
                
    def _get_weighted_heights(self, input_z, weights):
        '''
        This function itend to multiply the input_z by weights with displacemet in row and col.
        padding is 'same'.
        input_z - tensor [None, rows, cols]
        weights - tensor [None, rows, cols, window_size^2]
        '''
        with tf.name_scope('weighting_heights'):
            padded_z = get_padded_z(input_z, self.win_size)
            weighted_z = tf.zeros(tf.shape(input_z), name = 'weighted_z')
        
            for row_dis in range(self.win_size):
                for col_dis in range(self.win_size):
                    weight_index = self.win_size*row_dis + col_dis
                    weighted_z += get_weighted_heights_displacment(padded_z, weights[:, :, :, weight_index], row_dis, col_dis)
            
            return weighted_z
        
class HeightAverager(HeightWeighterBase):
    
    def __init__(self, win_size, min_height = 0, max_height = 25):
        '''
        Window size should be 1,3,5,7,....
        '''
        HeightWeighterBase.__init__(self, win_size, min_height, max_height)
        
    def get_averaged_heights(self, input_z, weights):
        
        with tf.name_scope('averaging_heights'):
            weighted_z = self._get_weighted_heights(input_z, weights)
            
            normed_heights = self.__get_normed_heights(weights, weighted_z)
            return normed_heights
        
    def __get_normed_heights(self, weights, weighted_z):
        
        with tf.name_scope('normalizing_heights'):
            norm_weights = get_norm_of_weights(weights)
            normed_heights = weighted_z/norm_weights
            normed_heights = tf.maximum(normed_heights, self.min_height)
            normed_heights = tf.minimum(normed_heights, self.max_height)
            return normed_heights 
        
class HeightDeviator(HeightWeighterBase):
    
    def __init__(self, win_size, num_rows, num_cols, min_height = 0, max_height = 25):
        '''
        Window size should be 1,3,5,7,....
        '''
        HeightWeighterBase.__init__(self, win_size, min_height, max_height)
        self.shape = [-1, num_rows, num_cols, 1]
        
    def get_averaged_heights(self, input_z, weights):
        
        with tf.name_scope('averaging_heights'):
            centered_weights = self.__get_centered_weights(weights)
            weighted_z = self._get_weighted_heights(input_z, centered_weights) + input_z
            weighted_z = tf.maximum(weighted_z, self.min_height)
            weighted_z = tf.minimum(weighted_z, self.max_height)
            return weighted_z
        
    def __get_centered_weights(self, weights):
        
        with tf.name_scope('centering_heights'):
            average_weights = tf.reduce_mean(weights, axis = 3, name = 'average_of_weights')
            average_weights = tf.reshape(average_weights, self.shape)
            centered_weights = weights - average_weights
            return centered_weights 

def get_padded_z(input_z, win_size):
    '''
    Padding z so we could convientley choose slices.
    '''
    with tf.name_scope('padding_heights'):
        pad_size = int(win_size/2)
        padding = tf.constant([[0, 0,], [pad_size, pad_size,], [pad_size, pad_size]])
        padded_z = tf.pad(input_z, padding, "SYMMETRIC")
        return padded_z
    
def get_weighted_heights_displacment(padded_z, weights, row_dis, col_dis):
    '''
    This function itend to multiply the input_z by weights with displacemet in row and col.
    padding is 'same'.
    padded_z - tensor [None, rows + win_size, cols + win_size]
    weights - tensor [None, rows, cols]
    row_dis - integer in range(win_size)
    col_dis - integer in range(win_size)
    '''
    name_scp = 'weighting_heights_row_dis_{}_col_dis_{}'.format(row_dis, col_dis)
    with tf.name_scope(name_scp):
        shape = tf.shape(weights)
        sliced_z = tf.slice(padded_z, [0, row_dis, col_dis], shape)
        weighted_z = tf.multiply(weights, sliced_z, \
                     name = 'weighting_z_row_dis_{}_col_dis_{}'.format(row_dis, col_dis))
        return weighted_z

def is_square(integer):
        
    root = math.sqrt(float(integer))
    return int(root + 0.5) ** 2 == integer