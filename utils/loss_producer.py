# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:48:56 2018

@author: mmoshe
"""

import tensorflow as tf
import utils.cost_functions as cost_functions

class LossTrainingProducerBase:

    def __init__(self, params):
        self.params = params
        self.num_rows, self.num_cols = self.params['image_shape']
        self.cost_params = self.params.get('cost_params', dict())

class LossTrainingProducer(LossTrainingProducerBase):
    
    def __init__(self, params):

        LossTrainingProducerBase.__init__(self, params)
        
    def get_loss(self, graph_dict):
        with tf.name_scope('loss_and_train'):
            residuals, stl_z = self.__get_residuals(graph_dict['z_predicted'])
            loss_func, accuracy = self.__get_loss(residuals)
                
            with tf.name_scope('train'):
                learning_rate = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')
                train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_func)
            
            return {'accuracy': accuracy, 'train_step': train_step, 
                    'loss_func': loss_func, 'y': stl_z, 'learning_rate': learning_rate}

    def __get_residuals(self, predicted_z):
        with tf.name_scope('producing_residuals'):
            stl_z = tf.placeholder(tf.float32, [None, self.num_rows, self.num_cols], name='true_z')
            residuals = tf.subtract(predicted_z, stl_z, name='residuals')
            return residuals, stl_z

    def __get_loss(self, residuals):
        with tf.name_scope('loss_layer'):
            loss_func = getattr(cost_functions, self.params['cost'])(residuals, self.cost_params)
            accuracy = loss_func / tf.cast(tf.size(residuals), tf.float32)
            tf.summary.scalar('accuracy', accuracy)

            return loss_func, accuracy

class LossTrainingProducerFiniteRange(LossTrainingProducerBase):

    def __init__(self, params):
        LossTrainingProducerBase.__init__(self, params)
        self.range_size_pix = params['range_size_pix']

    def get_loss(self, graph_dict):
        with tf.name_scope('loss_and_train'):
            residuals, stl_z, num_non_zero = self._get_residuals(graph_dict['z_predicted'], graph_dict['input'])
            loss_func, accuracy = self._get_loss(residuals, num_non_zero)

            with tf.name_scope('train'):
                learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
                train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_func)

            return {'accuracy': accuracy, 'train_step': train_step,
                    'loss_func': loss_func, 'y': stl_z, 'learning_rate': learning_rate}

    def _get_residuals(self, predicted_z, original_z):
        with tf.name_scope('producing_residuals'):
            near_non_zero = get_mask_near_non_zeros(original_z, self.range_size_pix)
            num_non_zero = tf.reduce_sum(near_non_zero)
            stl_z = tf.placeholder(tf.float32, [None, self.num_rows, self.num_cols], name='true_z')
            tot_residuals = tf.subtract(predicted_z, stl_z, name='total_residuals')
            residuals = tf.where(near_non_zero > 0, tot_residuals, tf.zeros(tf.shape(tot_residuals)))
            return residuals, stl_z, num_non_zero

    def _get_loss(self, residuals, num_non_zeros):

        with tf.name_scope('loss_layer'):
            loss_func = getattr(cost_functions, self.params['cost'])(residuals, self.cost_params)
            accuracy = loss_func / num_non_zeros
            tf.summary.scalar('accuracy', accuracy)

            return loss_func, accuracy

class LossTrainingProducerFiniteRangeMovingTissue(LossTrainingProducerFiniteRange):

    def __init__(self, params):
        LossTrainingProducerFiniteRange.__init__(self, params)
        self.moving_tissue_slope = params['cost_params']['moving_tissue_slope']

    def get_loss(self, graph_dict):
        with tf.name_scope('loss_and_train'):
            residuals, stl_z, num_non_zero = self._get_residuals(graph_dict['z_predicted'], graph_dict['input'])
            moving_tissue_loss = self.__get_moving_tissue_loss(residuals)
            plain_loss, accuracy = self._get_loss(residuals, num_non_zero)
            loss_func = plain_loss + moving_tissue_loss

            with tf.name_scope('train'):
                learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
                train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_func)

            return {'accuracy': accuracy, 'train_step': train_step,
                    'loss_func': loss_func, 'y': stl_z, 'learning_rate': learning_rate}

    def __get_moving_tissue_loss(self, residuals):
        with tf.name_scope('moving_tissue_loss'):
            mtl = tf.reduce_sum(tf.abs(residuals))
            return self.moving_tissue_slope*mtl

def get_mask_near_non_zeros(image, pool_size):
    from utils.utils import reshape_image
    with tf.name_scope('masking_range'):
        pre_shape = image.get_shape()
        ps = [-1, pre_shape[1].value, pre_shape[2].value]
        near_non_zero = tf.layers.max_pooling2d(inputs = reshape_image(image), pool_size = pool_size, strides = 1, padding = 'same')
        return tf.reshape(tf.sign(near_non_zero), ps)