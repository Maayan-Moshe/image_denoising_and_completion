#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:53:24 2018

@author: deeplearning
"""

import tensorflow as tf
from .utils import get_summary_path

class Trainer:
    
    def __init__(self, sess, components, params):
        
        self.graph_p = components['graph']
        self.learning_rate = params['initial_learning_rate']
        self.min_learning_rate = params['minimum_learning_rate']
        self.dat_feeder = components['train feeder']
        self.merged = components['summary merge']
        self.sess = sess
        self.__init_summary(params)
        
    def train_batch(self, index, check):
        
        f_dict = self.__feed_dict()
        if check:
            self.__check_and_summary(f_dict, index)
        self.graph_p['train_step'].run(feed_dict = f_dict)
        
    def reduce_learning_rate(self):
        
        print('Previous learning rate {}, new leraning rate {}.'.format(self.learning_rate, self.learning_rate/2))
        self.learning_rate /= 2

    def should_continue(self):

        if self.learning_rate > self.min_learning_rate:
            return True
        return False
        
    def __init_summary(self, params):
        
        sum_path = get_summary_path(params['summaries_dir'], params['summary_name'], kind = 'train')
        self.train_writer = tf.summary.FileWriter(sum_path, self.sess.graph)
    
    def __feed_dict(self):
        batch = self.dat_feeder.next_batch()
        f_dict = {self.graph_p['input']: batch['input'], 
                  self.graph_p['y']: batch['truth'], 
                  self.graph_p['learning_rate']: self.learning_rate}
        return f_dict
    
    def __check_and_summary(self, f_dict, index):
        
        summary, acc, loss = self.sess.run([self.merged, self.graph_p['accuracy'], self.graph_p['loss_func']], feed_dict=f_dict)
        self.train_writer.add_summary(summary, index)
        print('step {}, training: accuracy {}, loss {}'.format(index, acc, loss))