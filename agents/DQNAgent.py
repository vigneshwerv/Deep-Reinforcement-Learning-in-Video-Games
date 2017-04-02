from BaseAgent import BaseAgent

import tensorflow as tf

class DQNAgent(BaseAgent):

    def __init__(self, num_actions):
        self.learning_rate = 0.00025
        self.gradient_momentum = 0.95
        self.sq_gradient_momentum = 0.95
        self.min_sq_gradient = 0.01
        self.num_actions = num_actions
        self.x = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])
        self.action = tf.placeholder(tf.float32, shape=[None, self.num_actions])
        self.y = tf.placeholder(tf.float32, shape=[None])
        self.__create_training_network__()
        self.__create_playing_network__()
        self.session = tf.Session()
        initialize_variables = tf.global_variables_initializer()
        self.session.run(initialize_variables)
        self.copy_weights()

    def __conv2d__(self, input_tensor, filters, stride):
        return tf.nn.conv2d(input_tensor, filters,
                strides=[1, stride, stride, 1],
                padding="SAME")

    def __create_weights__(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def __create_biases__(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def __create_playing_network__(self):
        # Playing network variables
        self.pconv_w_1 = self.__create_weights__([8, 8, 4, 32])
        self.pconv_b_1 = self.__create_biases__([32])
        self.pconv_1 = tf.nn.relu(
                self.__conv2d__(self.x, self.pconv_w_1, 4) + self.pconv_b_1)

        self.pconv_w_2 = self.__create_weights__([4, 4, 32, 64])
        self.pconv_b_2 = self.__create_biases__([64])
        self.pconv_2 = tf.nn.relu(
                self.__conv2d__(self.pconv_1, self.pconv_w_2, 2) + self.pconv_b_2)

        self.pconv_w_3 = self.__create_weights__([3, 3, 64, 64])
        self.pconv_b_3 = self.__create_biases__([64])
        self.pconv_3 = tf.nn.relu(
                self.__conv2d__(self.pconv_2, self.pconv_w_3, 1) + self.pconv_b_3)

        self.pconv_flat = tf.reshape(self.pconv_3, [-1, 7744])

        self.pfc_w_1 = self.__create_weights__([7744, 512])
        self.pfc_b_1 = self.__create_biases__([512])
        self.pfc_1 = tf.nn.relu(tf.matmul(self.pconv_flat, self.pfc_w_1) + self.pfc_b_1)

        self.pout_w_2 = self.__create_weights__([512, self.num_actions])
        self.pout_b_2 = self.__create_biases__([self.num_actions])
        self.poutput = tf.matmul(self.pfc_1, self.pout_w_2) + self.pout_b_2
        
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
        self.tconv_w_1 = self.__create_weights__([8, 8, 4, 32])
        self.tconv_b_1 = self.__create_biases__([32])
        self.tconv_1 = tf.nn.relu(
                self.__conv2d__(self.x, self.tconv_w_1, 4) + self.tconv_b_1)

        self.tconv_w_2 = self.__create_weights__([4, 4, 32, 64])
        self.tconv_b_2 = self.__create_biases__([64])
        self.tconv_2 = tf.nn.relu(
                self.__conv2d__(self.tconv_1, self.tconv_w_2, 2) + self.tconv_b_2)

        self.tconv_w_3 = self.__create_weights__([3, 3, 64, 64])
        self.tconv_b_3 = self.__create_biases__([64])
        self.tconv_3 = tf.nn.relu(
                self.__conv2d__(self.tconv_2, self.tconv_w_3, 1) + self.tconv_b_3)

        self.tconv_flat = tf.reshape(self.tconv_3, [-1, 7744])

        self.tfc_w_1 = self.__create_weights__([7744, 512])
        self.tfc_b_1 = self.__create_biases__([512])
        self.tfc_1 = tf.nn.relu(tf.matmul(self.tconv_flat, self.tfc_w_1) + self.tfc_b_1)

        self.tout_w_2 = self.__create_weights__([512, self.num_actions])
        self.tout_b_2 = self.__create_biases__([self.num_actions])
        self.toutput = tf.matmul(self.tfc_1, self.tout_w_2) + self.tout_b_2

        self.readout = tf.reduce_sum(tf.multiply(self.toutput, self.action), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y - self.readout))
        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

    def copy_weights(self):
        self.session.run([self.pconv_w1_assign, self.pconv_b1_assign,
                          self.pconv_w2_assign, self.pconv_b2_assign,
                          self.pconv_w3_assign, self.pconv_b3_assign,
                          self.pfc_w_1_assign, self.pfc_b_1_assign,
                          self.pout_w_2_assign, self.pout_b_2_assign])
        return

    def predict(self, minibatch):
        return self.session.run(self.poutput, feed_dict={self.x: minibatch})

    def train(self, x_values, y_values, actions):
        cost, _ = self.session.run([self.cost, self.train_step], feed_dict={
                                                self.x: x_values,
                                                self.y: y_values,
                                                self.action: actions
                                               })
        return cost
