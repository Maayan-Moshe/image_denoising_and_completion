
import gray_image_reconstruction.gray_image_reducer as gray_image_reducer
import gray_image_reconstruction.data_expander_averager as data_expander_averager
import tensorflow as tf

class ReducerExpander:
    
    def __init__(self, params):
        
        self.params = params
        self.reducer_class = getattr(gray_image_reducer, params['reduction']['reducer'])
        self.expander_class = getattr(data_expander_averager, params['expansion']['expander'])
        
    def reduce_expand_average_image(self, pre_data = {'image': None, 'weights': None}):
        pre_shape = pre_data['image'].get_shape()
        if pre_shape[1].value < 3 or pre_shape[2].value < 3:
                return pre_data
        reduced_data, pre_reduced_dat = self.__reduce_image(pre_data)
        average_data = self.__get_expanded_image(reduced_data, pre_reduced_dat, pre_data)
        return average_data
        
    def __reduce_image(self, pre_data):
        pre_shape = pre_data['image'].get_shape()
        ps = [-1, pre_shape[1].value, pre_shape[2].value, 1]
        
        pre_reduced_dat = self.__get_reduced_image(pre_data, ps[1], ps[2])
        average_data = self.reduce_expand_average_image(pre_reduced_dat)
        return average_data, pre_reduced_dat
        
    def __get_expanded_image(self, reduced_data, pre_reduced_dat, pre_data):
        ps = pre_data['image'].get_shape()
        with tf.name_scope('expanding_image_{}x{}'.format(ps[1].value, ps[2].value)):
             expander_inst = self.expander_class(self.params['expansion'])
             average_data = expander_inst.get_gray_image(reduced_data, pre_reduced_dat, pre_data)
             return average_data
        
    def __get_reduced_image(self, pre_data, num_rows, num_cols):
        with tf.name_scope('reducing_image_{}x{}'.format(num_rows, num_cols)):
            reducer_inst = self.reducer_class(num_rows, num_cols, self.params['reduction'])
            reduced_dat = reducer_inst.get_reduced(pre_data)
            return reduced_dat 
    

        