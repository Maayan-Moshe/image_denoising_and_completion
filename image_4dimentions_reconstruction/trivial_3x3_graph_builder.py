import tensorflow as tf
import utils.cost_functions as cost_functions

PARAMS = {'image shape': (155, 208),
          'cost': 'l1_cost',
          'data_path': '/sample_data/image_4dimentions_reconstruction/20171224141532/4d_images.npy'}

def prepare_graph(params = {'image shape': (155, 208), 'cost': 'l2_cost'}):
    
    num_rows, num_cols = params['image shape']
    with tf.name_scope('simple_averaging'):
        y_conv, x = conv_layer(num_rows, num_cols)
    
    y_ = tf.placeholder(tf.float32, [None, num_rows, num_cols])
    y_image = tf.reshape(y_, [-1, num_rows, num_cols, 1])
    
    residuals = y_image - y_conv
        
    loss_func = getattr(cost_functions, params['cost'])(residuals)
    
    learning_rate = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')
    
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_func)
    
    accuracy = loss_func/tf.cast(tf.size(residuals), tf.float32)
    tf.summary.scalar('accuracy', accuracy)
    return {'accuracy': accuracy, 'train_step': train_step, 
            'loss_func': loss_func, 'input': x, 'y': y_, 
            'z_predicted': y_conv, 'learning_rate': learning_rate}
    
def conv_layer(num_rows, num_cols):
    
    W_conv = weight_variable([3, 3, 1, 1], name = 'weight_conv')
    b_conv = bias_variable([1], name = 'bias')
    
    tf.summary.scalar('bias', b_conv[0])
    
    x = tf.placeholder(tf.float32, [None, num_rows, num_cols])
    x_image = tf.reshape(x, [-1, num_rows, num_cols, 1])
    
    y_conv = conv2d(x_image, W_conv) + b_conv
    return y_conv, x
    
def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name = name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name = name)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')