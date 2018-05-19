
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
        
class PredictedTensorPrinter:
    
    def __init__(self, params):
        from gray_image_reconstruction.multiscale_graph_builder import prepare_graph
        import data_preparation.DataFeeders as data_feeders
        
        self.graph_p = prepare_graph(params['graph_params'])
        self.saver = tf.train.Saver()
        state_params = params["state_params"]
        self.state_fname = os.path.join(state_params["state_folder"], state_params["state_fname"])
        self.feeder = getattr(data_feeders, params['data_params']['feeder'])(**params['data_params'])

    def add_expected_heights_and_save(self, out_path):
        
        with tf.Session() as sess:
            self.__perform(sess, out_path, self.state_fname)
            
    def __perform(self, sess, out_path, state_fname):
        
        self.saver.restore(sess, state_fname)
        feed_dict = self.feeder.get_all_data()
        perd_z_tf = self.graph_p['z_predicted'].eval(feed_dict={self.graph_p['input']: feed_dict['input']})
        feed_dict['predicted z'] = perd_z_tf
        np.save(out_path, feed_dict)

def plot_result(results, image_num):
    import matplotlib.pyplot as plt

    input_img = results['input'][image_num]
    predicted_img = results['predicted z'][image_num]
    true_img = results['truth'][image_num]
    fig, ((ax_in, ax_prd, ax_tr)) = plt.subplots(3, 1)
    ax_in.imshow(input_img, 'gray')
    ax_in.set(title='semilogy')

if __name__ == '__main__':
    import json

    with open(r'results_configuration.json') as config_file:
        params = json.load(config_file)
    out_path = r'results\real_dat_19_May_2018_12_24.npy'
    PredictedTensorPrinter(params).add_expected_heights_and_save(out_path)

#    VariablePrinter().print_vars()
