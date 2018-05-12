
import tensorflow as tf
import os
import numpy as np

STATE_PARAMS = {'state_folder': r'path_to_state_files\saved_models\image_multi_scale',
                'data_path': r"path_to_data\s0_s1_validation.npy",
                'state_fname': 'real_data_multiscale_12_Mar_2018_12_23.ckpt',
                'cropping': 0}

DATA_PATH = '/sample_data/image_4dimentions_reconstruction/20171224141532/predicted_z.npy'

class VariablePrinter:
    
    def __init__(self, state_fname = 'model_3x3_2017-12-27 22:01:35.636482.ckpt'):
        
        self.state_fname = state_fname
        self.weight = tf.get_variable('weight_conv', shape=[3,3,1,1])
        self.bias = tf.get_variable('bias', shape=[1]) 
        self.tot_weight = tf.reduce_sum(self.weight)
        self.saver = tf.train.Saver()
        
    def print_vars(self):
        
        with tf.Session() as sess:
            self.__perform_print(sess)
            
    def __perform_print(self, sess):
        self.saver.restore(sess, os.path.join(STATE_PARAMS['state_folder'], self.state_fname))
        weight_result = np.array(self.weight.eval()).reshape((3,3))
        print("wieghts conv : %s" % weight_result)
        print("bias : %s" % self.bias.eval())
        print("total weight : %s" % self.tot_weight.eval())
        
class PredictedHeightMapPrinter:
    
    def __init__(self, state_params, graph_params):
        from image_4dimentions_reconstruction.trivial_3x3_graph_builder import prepare_graph

        self.shape = graph_params['image_shape']
        self.cost_type = graph_params['cost']
        self.graph_p = prepare_graph(graph_params)
        self.saver = tf.train.Saver()
        self.sparams = state_params
        
    def add_expected_heights_and_save(self):
        
        self.data = np.load(self.sparams['data_path'], encoding = 'latin1').tolist()
        with tf.Session() as sess:
            self.__perform(sess)
            
    def __perform(self, sess):

        state_fname = os.path.join(self.sparams['state_folder'], self.sparams['state_fname'])
        self.saver.restore(sess, state_fname)
        for scan_dat in self.data.itervalues():
            self.__add_predicted_to_map(scan_dat)
        np.save(self.sparams['data_path'], self.data)
    
    def __add_predicted_to_map(self, scan_dat):

        orig_height = scan_dat['z'].reshape((1, self.shape[0], self.shape[1]))
        perd_z_tf = self.graph_p['z_predicted'].eval(feed_dict={self.graph_p['input']: orig_height})
        out_field_name = 'z_' + self.cost_type
        scan_dat[out_field_name] = np.array(perd_z_tf).reshape(self.shape)
        
class PredictedTensorPrinter:
    
    def __init__(self, state_params, graph_params):
        from gray_image_reconstruction.multiscale_graph_builder import prepare_graph
        from data_preparation.DataFeeders import OneFileValidatorFeeder
        
        self.graph_p = prepare_graph(graph_params)
        self.saver = tf.train.Saver()
        self.feeder = OneFileValidatorFeeder(state_params['data_path'], state_params['cropping'])

    def add_expected_heights_and_save(self, state_fname, out_path):
        
        with tf.Session() as sess:
            self.__perform(sess, out_path, state_fname)
            
    def __perform(self, sess, out_path, state_fname):
        
        self.saver.restore(sess, state_fname)
        feed_dict = self.feeder.get_all_data()
        perd_z_tf = self.graph_p['z_predicted'].eval(feed_dict={self.graph_p['input']: feed_dict['input']})
        feed_dict['predicted z'] = perd_z_tf
        np.save(out_path, feed_dict)

if __name__ == '__main__':
    
    graph_params = {'image_shape': (178, 233), 'cost': 'l2_cost',
                    'reduction': {
                        'reducer': 'HeightMapReducerFiller',
                        'regularization': {'kernel': 5e-1, 'bias': 5e-1}},
                    'expansion': {
                        'expander': 'DataExpanderAveragerAdditioner',
                        'regularization': {'kernel': 5e-1, 'bias': 5e-1}},
                    'loss_producer': 'LossTrainingProducerFiniteRange',
                    'range_size_pix': 11,
                    'max_z_mm': 25, 'min_z_mm': 0}
    state_path = os.path.join(STATE_PARAMS['state_folder'], STATE_PARAMS['state_fname'])
    out_path = r'path_to_state_files\results\real_dat_12_Mar_2018_12_23.npy'
    PredictedTensorPrinter(STATE_PARAMS, graph_params).add_expected_heights_and_save(state_path, out_path)

#    VariablePrinter().print_vars()
