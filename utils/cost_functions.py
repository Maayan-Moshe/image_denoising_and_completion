#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 12:51:12 2017

@author: deeplearning
"""

import tensorflow as tf

def l1_cost(residuals, params = None):
    
    res_abs = tf.abs(residuals)
    loss_func = tf.reduce_sum(res_abs)    
    return loss_func

def l2_cost(residuals, params = None):
    
    res_sqr = tf.square(residuals)
    loss_func = tf.reduce_sum(res_sqr)    
    return loss_func

def huber_cost(residuals, params = None):
    
    res_abs = tf.abs(residuals)
    tensor_cost = tf.where(res_abs < 1, tf.square(res_abs)/2, res_abs - 0.5)
    loss_func = tf.reduce_sum(tensor_cost)
    return loss_func
    
def derivative_cost(residuals, params = None):
    
    dx = residuals[:, 1: , :] - residuals[:, :-1 , :]
    dy = residuals[:, : , 1:] - residuals[:, : , :-1]
    loss_func = tf.reduce_sum(tf.square(dx)) + tf.reduce_sum(tf.square(dy))
    return loss_func

def derivative_huber_cost(residuals, params = None):

    dx = residuals[:, 1:, :] - residuals[:, :-1, :]
    dy = residuals[:, :, 1:] - residuals[:, :, :-1]
    loss_func = huber_cost(dx) + huber_cost(dy)
    return loss_func

def derivative_abs_cost(residuals, params = None):
    dx = residuals[:, 1:, :] - residuals[:, :-1, :]
    dy = residuals[:, :, 1:] - residuals[:, :, :-1]
    loss_func = tf.reduce_sum(tf.abs(dx)) + tf.reduce_sum(tf.abs(dy))
    return loss_func

def huber_plus_derivative_cost(residuals, params):

    derivative_strength = params['derivative_strength']
    loss_func = huber_cost(residuals) + derivative_strength*derivative_cost(residuals)
    return loss_func

def huber_plus_huber_derivative_cost(residuals, params):

    derivative_strength = params['derivative_strength']
    loss_func = huber_cost(residuals) + derivative_strength*derivative_huber_cost(residuals)
    return loss_func

def huber_plus_abs_derivative_cost(residuals, params):

    derivative_strength = params['derivative_strength']
    loss_func = huber_cost(residuals) + derivative_strength * derivative_abs_cost(residuals)
    return loss_func

def square_plus_abs_derivative(residuals, params):

    derivative_strength = params['derivative_strength']
    square_cost = tf.reduce_sum(tf.square(residuals))
    loss_func = square_cost + derivative_strength * derivative_abs_cost(residuals)
    return loss_func