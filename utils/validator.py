#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:20:45 2018

@author: deeplearning
"""
import tensorflow as tf
from .utils import get_summary_path

class Validator:
    
    def __init__(self, sess, components, params):
        
        self.graph_p = components['graph']
        self.dat_feeder = components['validation feeder']
        self.merged = components['summary merge']
        self.sess = sess
        sum_path = get_summary_path(params['summaries_dir'], params['summary_name'], kind = 'validation')
        self.validation_writer = tf.summary.FileWriter(sum_path, self.sess.graph)
    
    def validate(self, index):
        
        f_dict = self.__feed_dict()
        summary, acc, loss = self.sess.run([self.merged, self.graph_p['accuracy'], self.graph_p['loss_func']], feed_dict=f_dict)
        self.validation_writer.add_summary(summary, index)
        print('step {}, validation: accuracy {}, loss {}'.format(index, acc, loss))
        return loss
        
    def __feed_dict(self):
        
        batch = self.dat_feeder.get_all_data()
        f_dict = {self.graph_p['input']: batch['input'], 
                  self.graph_p['y']: batch['truth']}
        return f_dict