import numpy as np
import tensorflow as tf
from .reduce_expand_layer import ReducerExpander
import utils.loss_producer as loss_producer

PARAMS = {'image_shape': (155, 208), 
          'cost': 'huber_plus_derivative_cost', 
          'cost_params': {'derivative_strength': 4e-1,
                        'moving_tissue_slope': 2e0},
          'loss_producer': 'LossTrainingProducerFiniteRangeMovingTissue',
          'range_size_pix': 11,
          'max_z_mm': 25, 'min_z_mm': 0,
          'reduction': {
              'reducer': 'HeightMapReducerFiller',
              'regularization': {'kernel': 0.5, 'bias': 0.5}},
          'expansion': {
                        'expander': 'DataExpanderAveragerAdditioner',
                        'regularization': {'kernel': 5e-1, 'bias': 5e-1}}
          }

def prepare_graph(params = PARAMS):
    import itertools

    graph_dict = prepare_graph_without_loss(params)
    loss_dict = getattr(loss_producer, params['loss_producer'])(params).get_loss(graph_dict)
    tot_dict = dict(itertools.chain(graph_dict.items(), loss_dict.items()))
    return tot_dict

def prepare_graph_without_loss(params = PARAMS):
    
    num_rows, num_cols = params['image_shape']
    input_layer = tf.placeholder(tf.float32, [None, num_rows, num_cols], name = 'input')
    reducer_expander = ReducerExpander(params)
    predicted_data = reducer_expander.reduce_expand_average_image({'image': input_layer})
    clipped_image = clip_image(predicted_data['image'], params['min_z_mm'], params['max_z_mm'])
    predicted_image = tf.reshape(clipped_image, [-1, num_rows, num_cols])
    graph_dict = {'z_predicted': predicted_image, 'input': input_layer}
    return graph_dict  

def clip_image(image, min_z, max_z):
    with tf.name_scope('clipping_height_map'):
        clipped_image = tf.maximum(image, min_z)
        clipped_image = tf.minimum(image, max_z)
        return clipped_image
    
if __name__ == '__main__':
    graph_dict = prepare_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(graph_dict['z_predicted'], feed_dict={graph_dict['input']: np.ones((3, 155, 208, 1))})
        writer = tf.summary.FileWriter('/home/deeplearning/Documents/results/summaries_images', graph=sess.graph)
