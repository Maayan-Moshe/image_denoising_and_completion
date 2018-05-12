# -*- coding: utf-8 -*-
"""
Created on Tue Feb 06 08:27:53 2018

@author: mmoshe
"""
import json

import tensorflow as tf
import os

def get_norm_of_weights(weights):
    '''
    This function return the norm of the weights. Input [:, num_rows, num_cols, n] output [:, num_ros, num_cols].
    We wish to have the property that constant heights yield same height.
    '''
    with tf.name_scope('normalizing_weights'):
        sum_weights = tf.reduce_sum(weights, axis = 3)
        norm_weights = tf.maximum(sum_weights, 1, name = 'norm_weights')
        return norm_weights
        
def get_one_sum_kernel(kernel_size, num_inputs, num_outputs):
    '''
    This function is inteded to return kernel variable of the size, [kernel_size, kernel_size, num_inputs, num_outputs].
    The sum of kernel entries for each chunnel i.e. np.sum(ker[:, :, :, i]) = 1, is kept to be one.
    '''
    with tf.name_scope('zero_mean_kernel'):
        slice_size = kernel_size*kernel_size*num_inputs
        kernel_var = tf.get_variable(name = 'expansion_kernel', shape = [slice_size, num_outputs], initializer = tf.contrib.layers.xavier_initializer())
        ker_reduced_mean = tf.reduce_mean(kernel_var, axis = 0)
        kernel_var += 1./slice_size - ker_reduced_mean
        kernel = tf.reshape(kernel_var, [kernel_size, kernel_size, num_inputs, num_outputs])
        return kernel
        
def get_normalized_weights(weights):
    with tf.name_scope('normalizing_weights'):
        weights_norm = get_norm_of_weights(weights)
        w_shape = weights.get_shape()
        ws = [-1, w_shape[1].value, w_shape[2].value, 1]
        normed_weights = weights/tf.reshape(weights_norm, ws)
        return normed_weights

def get_checkpoint_dict_for_new_namespace(checkpoint, namespace):
    from tensorflow.python import pywrap_tensorflow
    
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint)
    return {
        "{}/{}".format(namespace, k): k
        for k in reader.get_variable_to_shape_map()
    }

def get_summary_path(folder, fname, kind = 'train'):
    import os
    from time import gmtime, strftime
    
    o_fold = os.path.join(folder, kind)
    time_str = strftime("%d_%b_%Y_%H_%M", gmtime())
    summ_path = os.path.join(o_fold,'{}_{}'.format(fname, time_str))
    if not os.path.exists(summ_path):
        os.makedirs(summ_path)
    return summ_path
    
def stack_input(previous_data):
    with tf.name_scope('stacking_input'):
        prev_tensors = [reshape_image(value) for value in previous_data.values()]
        stacked_input = tf.concat(prev_tensors, axis = 3, name = 'stacked_input')
                    
        return stacked_input
    
def reshape_image(pre_image):
    with tf.name_scope('reshaping_image'):
        pre_shape = pre_image.get_shape()
        if len(pre_shape) == 4:
            return pre_image
        ps = [-1, pre_shape[1].value, pre_shape[2].value, 1]
        image = tf.reshape(pre_image, ps)
        return image


def refactor_paths(param_dictionary, keys_to_refactor=None, makedirs=True):
    """
    expands all ~ o homedir for the keys in the keys_to_refactor dict
    :param param_dictionary: parameters
    :param keys_to_refactor: keys which to expand
    :param makedirs: also create the folders
    :return:
    """
    if keys_to_refactor is None:
        keys_to_refactor = {
            "train_params": ["state_fname", "state_folder", "summaries_dir"],
            "data_params": ["folder", "data_path"],
        }

    for k, vals in keys_to_refactor.items():
        for v in vals:
            param_dictionary[k][v] = os.path.expanduser(param_dictionary[k][v])

    return param_dictionary


def load_params_from_json(json_fname):
    with open(json_fname) as f:
        params = json.load(f)

    params = refactor_paths(params)
    return params