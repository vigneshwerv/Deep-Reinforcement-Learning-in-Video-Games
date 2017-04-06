from BaseAgent import BaseAgent

import numpy as np
import tensorflow as tf

class DQNAgent(BaseAgent):

    def __init__(self, num_actions, discount_factor):
        self.learning_rate = 0.00025
        self.gradient_momentum = 0.95
        self.sq_gradient_momentum = 0.95
        self.min_sq_gradient = 0.01
        self.num_actions = int(num_actions)

        self.session = tf.Session()

        self.x = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])
        self.y = tf.placeholder(tf.float32, shape=[None, num_actions])
        self.global_step = tf.Variable(0, trainable=False)
        self.normalized_x = self.x / 255.0

        self.__create_training_network__()
        self.__create_playing_network__()

        initialize_variables = tf.global_variables_initializer()

        self.merged = tf.summary.merge_all()
        self.train_write = tf.summary.FileWriter('log/train', self.session.graph)

        self.session.run(initialize_variables)
        self.copy_weights(0, 0)

    def __conv2d__(self, input_tensor, output_dimension, filter_size, stride, name='conv'):
        initializer = tf.contrib.layers.xavier_initializer()
        stride = [1, stride[0], stride[1], 1]
        filter_size = [filter_size[0], filter_size[1], input_tensor.get_shape()[-1], output_dimension]
        with tf.variable_scope(name):
            w = tf.get_variable('w', filter_size, tf.float32, initializer=initializer)
            b = tf.get_variable('biases', [output_dimension], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.relu(tf.nn.conv2d(input_tensor, w, stride, padding='VALID') + b)
            return w, b, conv

    def __fc__(self, input_tensor, output_size, stddev=0.02, bias_start=0.0, name='fc'):
        shape = input_tensor.get_shape().as_list()
        with tf.variable_scope(name):
            w = tf.get_variable('Matrix', [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
            b = tf.get_variable('bias', [output_size], initializer=tf.constant_initializer(bias_start))
            out = tf.nn.relu(tf.matmul(input_tensor, w) + b)
            return w, b, out

    def __create_playing_network__(self):
        # Playing network variables
        self.pconv_w_1, self.pconv_b_1, self.pconv_1 = \
                self.__conv2d__(self.normalized_x, 32, [8, 8], [4, 4], name='p1')

        self.pconv_w_2, self.pconv_b_2, self.pconv_2 = \
                self.__conv2d__(self.pconv_1, 64, [4, 4], [2, 2], name='p2')

        self.pconv_w_3, self.pconv_b_3, self.pconv_3 = \
                self.__conv2d__(self.pconv_2, 64, [3, 3], [1, 1], name='p3')

        shape = self.pconv_3.get_shape().as_list()
        self.pconv_flat = tf.reshape(self.pconv_3, [-1, reduce(lambda x, y: x * y, shape[1:])])

        self.pfc_w_1, self.pfc_b_1, self.pfc_1 = \
                self.__fc__(self.pconv_flat, 512, name='p4')

        self.pout_w_2, self.pout_b_2, self.poutput = \
                self.__fc__(self.pfc_1, self.num_actions, name='p5')

        self.__create_assignment_ops__()

    def __create_assignment_ops__(self):
        # Creae graph ops for copying over weights to the playing network
        self.pconv_w1_assign = tf.assign(self.pconv_w_1, self.tconv_w_1)
        self.pconv_b1_assign = tf.assign(self.pconv_b_1, self.tconv_b_1)

        self.pconv_w2_assign = tf.assign(self.pconv_w_2, self.tconv_w_2)
        self.pconv_b2_assign = tf.assign(self.pconv_b_2, self.tconv_b_2)

        self.pconv_w3_assign = tf.assign(self.pconv_w_3, self.tconv_w_3)
        self.pconv_b3_assign = tf.assign(self.pconv_b_3, self.tconv_b_3)

        self.pfc_w_1_assign = tf.assign(self.pfc_w_1, self.tfc_w_1)
        self.pfc_b_1_assign = tf.assign(self.pfc_b_1, self.tfc_b_1)

        self.pout_w_2_assign = tf.assign(self.pout_w_2, self.tout_w_2)
        self.pout_b_2_assign = tf.assign(self.pout_b_2, self.tout_b_2)

    def __create_training_network__(self):
        # Training network variables
        self.tconv_w_1, self.tconv_b_1, self.tconv_1 = \
                self.__conv2d__(self.normalized_x, 32, [8, 8], [4, 4], name='t1')

        x_min = tf.reduce_min(self.tconv_w_1)
        x_max = tf.reduce_max(self.tconv_w_1)
        kernel_0_to_1 = (self.tconv_w_1 - x_min) / (x_max - x_min)
        kernel_transposed = tf.transpose(kernel_0_to_1, [3, 0, 1, 2])
        self.kernel_summary = tf.summary.image('conv1/filters', kernel_transposed, max_outputs=32)

        self.tconv_w_2, self.tconv_b_2, self.tconv_2 = \
                self.__conv2d__(self.tconv_1, 64, [4, 4], [2, 2], name='t2')

        self.tconv_w_3, self.tconv_b_3, self.tconv_3 = \
                self.__conv2d__(self.tconv_2, 64, [3, 3], [1, 1], name='t3')

        shape = self.tconv_3.get_shape().as_list()
        self.tconv_flat = tf.reshape(self.tconv_3, [-1, reduce(lambda x, y: x * y, shape[1:])])

        self.tfc_w_1, self.tfc_b_1, self.tfc_1 = \
                self.__fc__(self.tconv_flat, 512, name='t4')

        self.tout_w_2, self.tout_b_2, self.toutput = \
                self.__fc__(self.tfc_1, self.num_actions, name='t5')

        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                    100000, 0.96, staircase=True)

        self.cost = tf.reduce_mean(
                        tf.reduce_sum(
                            tf.squared_difference(
                                self.y, self.toutput), reduction_indices=1))
        self.train_step = tf.train.RMSPropOptimizer(learning_rate,
                                                    momentum=self.gradient_momentum,
                                                    epsilon=self.min_sq_gradient)\
                                .minimize(self.cost, global_step=self.global_step)

    def copy_weights(self, i, j):
        self.session.run([self.pconv_w1_assign, self.pconv_b1_assign,
                          self.pconv_w2_assign, self.pconv_b2_assign,
                          self.pconv_w3_assign, self.pconv_b3_assign,
                          self.pfc_w_1_assign, self.pfc_b_1_assign,
                          self.pout_w_2_assign, self.pout_b_2_assign])
        image_summary = self.session.run(self.kernel_summary)
        # self.train_write.add_summary(image_summary, i*j)
        return

    def predict(self, minibatch):
        if len(minibatch.shape) == 4:
            minibatch = np.transpose(minibatch, (0, 2, 3, 1))
        else:
            minibatch = np.transpose(minibatch, (1, 2, 0))
            minibatch = [minibatch]
        return self.session.run(self.poutput, feed_dict={self.x: minibatch})

    def train_predict(self, minibatch):
        if len(minibatch.shape) == 4:
            minibatch = np.transpose(minibatch, (0, 2, 3, 1))
        else:
            minibatch = np.transpose(minibatch, (1, 2, 0))
            minibatch = [minibatch]
        return self.session.run(self.toutput, feed_dict={self.x: minibatch})

    def train(self, x_values, y_values):
        x_values = np.transpose(x_values, (0, 2, 3, 1))
        cost, _ = self.session.run([self.cost, self.train_step], feed_dict={
                                                self.x: x_values,
                                                self.y: y_values
                                               })
        return cost
