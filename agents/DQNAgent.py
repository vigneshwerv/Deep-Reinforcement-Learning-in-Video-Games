from BaseAgent import BaseAgent

import math
import numpy as np
import tensorflow as tf

class DQNAgent(BaseAgent):

    def __init__(self, num_actions, discount_factor):

        self.learning_rate = 0.00025
        self.gradient_momentum = 0.95
        self.sq_gradient_momentum = 0.95
        self.min_sq_gradient = 0.01
        self.num_actions = int(num_actions)
        self.discount_factor = discount_factor

        self.error_clip = 1.0
        self.gradient_clip = 0

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

        self.screens = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])
        self.actions = tf.placeholder(tf.int32, shape=[None])
        self.rewards = tf.placeholder(tf.float32, shape=[None])
        self.terminals = tf.placeholder(tf.bool, shape=[None])
        self.next_screens = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])

        self.epsilon = tf.placeholder(tf.float32, shape=())
        self.avgscore = tf.placeholder(tf.float32, shape=())

        self.normalized_screens = self.screens / 255.0
        self.normalized_next_screens = self.next_screens / 255.0

        self.copy_ops = []
        self.q_values, self.target_q_values = self._create_network_()

        with tf.name_scope('summaries'):
            avg_qvalue = tf.reduce_mean(self.q_values)
            self.qvalue_summary = tf.summary.scalar('avg_qvalue', avg_qvalue)
            self.epsilon_summary = tf.summary.scalar('epsilon', self.epsilon)
            self.avgscore_summary = tf.summary.scalar('avg_score', self.avgscore)

        self.cost = self._create_error_gradient_ops_()
        self.train = self._create_optimizer_op_()

        initialize_variables = tf.global_variables_initializer()

        self.merged = tf.summary.merge_all()
        self.train_write = tf.summary.FileWriter('log/train', self.session.graph)

        self.session.run(initialize_variables)

    def _create_optimizer_op_(self):
        with tf.name_scope('optimizer'):
            optimizer = tf.train.RMSPropOptimizer(
                            self.learning_rate,
                            decay=self.gradient_momentum,
                            epsilon=self.min_sq_gradient)

            grads_and_vars = optimizer.compute_gradients(self.cost)
            gradients = [gradient[0] for gradient in grads_and_vars]
            params = [param[1] for param in grads_and_vars]

            if self.gradient_clip > 0:
                gradients = tf.clip_by_global_norm(gradients, self.gradient_clip)[0]
            return optimizer.apply_gradients(zip(gradients, params))

    def _create_error_gradient_ops_(self):
        # Get the Q-value for the action taken
        one_hot_actions = tf.stop_gradient(tf.one_hot(self.actions, self.num_actions))
        predicted_q_values = tf.reduce_sum(
                                tf.multiply(
                                    self.q_values, one_hot_actions),
                                reduction_indices=1)
        # Get the target Q-values (used for error)
        target_q_values = tf.reduce_max(self.target_q_values, reduction_indices=1)
        discount_factor = tf.constant(self.discount_factor)
        discount_q_values = tf.multiply(discount_factor, target_q_values)
        terminal_q_removed = tf.multiply(
                                (1.0 - tf.cast(self.terminals, tf.float32)),
                                discount_q_values)
        target_value = tf.stop_gradient(self.rewards + terminal_q_removed)
        # Calculate difference
        difference = tf.abs(predicted_q_values - target_value)

        if self.error_clip >= 0:
            clipped_errors = tf.clip_by_value(difference, -self.error_clip, self.error_clip)
            linear_part = difference - clipped_errors
            errors = (0.5 * tf.square(clipped_errors)) + (self.error_clip * linear_part)
        else:
            errors = (0.5 * tf.square(difference))

        return tf.reduce_sum(errors)

    def _get_weights_(self, shape, name):
        fan_in = np.prod(shape[0:-1])
        std = 1 / math.sqrt(fan_in)
        return tf.Variable(tf.random_uniform(shape, minval=(-std), maxval=std),
                           name=(name + '_weights'))
    def _get_biases_(self, shape, name):
        fan_in = np.prod(shape[0:-1])
        std = 1 / math.sqrt(fan_in)
        return tf.Variable(tf.random_uniform([shape[-1]], minval=(-std), maxval=std),
                           name=(name + '_biases'))

    def _create_conv_layer_(self,
                            policy_input_tensor,
                            target_input_tensor,
                            output_dimension,
                            filter_size,
                            stride,
                            name='conv'):
        stride = [1, stride[0], stride[1], 1]
        filter_size = [filter_size[0],
                       filter_size[1],
                       policy_input_tensor.get_shape().as_list()[-1],
                       output_dimension]
        with tf.variable_scope(name):
            w = self._get_weights_(filter_size, ('policy_' + name))
            b = self._get_biases_(filter_size, ('policy_' + name))
            target_w = tf.Variable(w.initialized_value(),
                                   trainable=False,
                                   name=('target_' + name + '_weights'))
            target_b = tf.Variable(b.initialized_value(),
                                   trainable=False,
                                   name=('target_' + name + '_biases'))

            self.copy_ops.append(target_w.assign(w))
            self.copy_ops.append(target_b.assign(b))

            policy_conv = tf.nn.relu(tf.nn.conv2d(policy_input_tensor, w, stride, padding='VALID') + b)
            target_conv = tf.nn.relu(tf.nn.conv2d(target_input_tensor,
                                                  target_w, stride,
                                                  padding='VALID') + target_b)
            return policy_conv, target_conv

    def _create_fc_layer_(self,
                          policy_input_tensor,
                          target_input_tensor,
                          output_size,
                          name='fc'):
        shape = policy_input_tensor.get_shape().as_list()
        with tf.variable_scope(name):
            w = self._get_weights_([shape[1], output_size], ('policy_' + name))
            b = self._get_biases_([shape[1], output_size], ('policy_' + name))
            target_w = tf.Variable(w.initialized_value(),
                                   trainable=False,
                                   name=('target_' + name + '_weights'))
            target_b = tf.Variable(b.initialized_value(),
                                   trainable=False,
                                   name=('target_' + name + '_biases'))

            self.copy_ops.append(target_w.assign(w))
            self.copy_ops.append(target_b.assign(b))

            policy_out = tf.nn.relu(tf.matmul(policy_input_tensor, w) + b)
            target_out = tf.nn.relu(tf.matmul(target_input_tensor, target_w) + target_b)
            return policy_out, target_out

    def _create_linear_layer_(self,
                             policy_input_tensor,
                             target_input_tensor,
                             output_size,
                             name='fc'):
        shape = policy_input_tensor.get_shape().as_list()
        with tf.variable_scope(name):
            w = self._get_weights_([shape[1], output_size], ('policy_' + name))
            b = self._get_biases_([shape[1], output_size], ('policy_' + name))
            target_w = tf.Variable(w.initialized_value(),
                                   trainable=False,
                                   name=('target_' + name + '_weights'))
            target_b = tf.Variable(b.initialized_value(),
                                   trainable=False,
                                   name=('target_' + name + '_biases'))

            self.copy_ops.append(target_w.assign(w))
            self.copy_ops.append(target_b.assign(b))

            policy_out = tf.matmul(policy_input_tensor, w) + b
            target_out = tf.matmul(target_input_tensor, target_w) + target_b
            return policy_out, target_out

    def _create_network_(self):
        policy_conv_1, target_conv_1 = self._create_conv_layer_(
                        self.normalized_screens,
                        self.normalized_next_screens,
                        32, [8, 8], [4, 4], name='conv1')
        policy_conv_2, target_conv_2 = self._create_conv_layer_(
                        policy_conv_1,
                        target_conv_1,
                        64, [4, 4], [2, 2], name='conv2')
        policy_conv_3, target_conv_3 = self._create_conv_layer_(
                        policy_conv_2,
                        target_conv_2,
                        64, [3, 3], [1, 1], name='conv3')

        shape = policy_conv_3.get_shape().as_list()
        policy_conv_3_flat = tf.reshape(policy_conv_3, [-1, reduce(lambda x, y: x * y, shape[1:])])
        target_conv_3_flat = tf.reshape(target_conv_3, [-1, reduce(lambda x, y: x * y, shape[1:])])

        policy_fc_1, target_fc_1 = self._create_fc_layer_(
                        policy_conv_3_flat,
                        target_conv_3_flat,
                        512, name='fc1')

        policy_out, target_out = self._create_linear_layer_(
                        policy_fc_1,
                        target_fc_1,
                        self.num_actions, name='linear')
        return policy_out, target_out

    def copy_weights(self):
        self.session.run(self.copy_ops)
        return

    def predict(self, minibatch):
        minibatch = np.transpose(minibatch, (0, 2, 3, 1))
        return self.session.run(self.q_values,
                                feed_dict={ self.screens: minibatch })

    def record_average_qvalue(self, minibatch, step, epsilon, avgscore):
        minibatch = np.transpose(minibatch, (0, 2, 3, 1))
        qvalue_summary, epsilon_summary, avgscore_summary  =\
                self.session.run([self.qvalue_summary,
                                  self.epsilon_summary,
                                  self.avgscore_summary],
                                  feed_dict={
                                      self.screens: minibatch,
                                      self.epsilon: epsilon,
                                      self.avgscore: avgscore
                                  })
        self.train_write.add_summary(qvalue_summary, step)
        self.train_write.add_summary(epsilon_summary, step)
        self.train_write.add_summary(avgscore_summary, step)

    def train_network(self, prestates, actions, rewards, terminals, poststates):
        prestates = np.transpose(prestates, (0, 2, 3, 1))
        poststates = np.transpose(poststates, (0, 2, 3, 1))
        cost, _ = self.session.run([self.cost, self.train],
                                   feed_dict={
                                    self.screens: prestates,
                                    self.actions: actions,
                                    self.rewards: rewards,
                                    self.terminals: terminals,
                                    self.next_screens: poststates
                                   })
        return cost
