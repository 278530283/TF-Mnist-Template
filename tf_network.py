import tensorflow as tf


class LeNet:

    def __init__(self):
        """
        Define some basic parameters here
        """
        # layer 1 parameters
        self.conv1_filter = [5, 5, 1, 6]
        self.conv1_out_channels = 6
        self.conv1_strides = [1, 1, 1, 1]
        # layer 2 parameters
        self.pool1_ksize = [1, 2, 2, 1]
        self.pool1_strides = [1, 2, 2, 1]
        # layer 3 parameters
        self.conv2_filter = [5, 5, 6, 16]
        self.conv2_out_channels = 16
        self.conv2_strides = [1, 1, 1, 1]
        # layer 4 parameters
        self.pool2_ksize = [1, 2, 2, 1]
        self.pool2_strides = [1, 2, 2, 1]
        # layer 5 parameters
        self.fc1_in_channels = self.conv2_filter[0] * self.conv2_filter[1] * self.conv2_out_channels
        self.fc1_out_channels = 120
        # layer 6 parameters
        self.fc2_out_channels = 84
        # layer 7 parameters
        self.final_out_channels = 10
        pass

    def net(self, feats):
        """
        Define network.
        You can use init_weight() and init_bias() function to init weight matrix,
        for example:
            conv1_W = self.init_weight((3, 3, 1, 6))
            conv1_b = self.init_bias(6)
        :param feats: input features
        :return: logits

        python版本3.5-3.7，tensorflow版本1.15.0
        下载sklearn库：`pip install scikit-learn
        """
        # layer 1
        # construct the conv1
        conv1_w = self.init_weight(self.conv1_filter)
        conv1_b = self.init_bias(self.conv1_out_channels)
        conv1 = tf.nn.conv2d(feats, conv1_w, strides=self.conv1_strides, padding='SAME') + conv1_b
        # layer 2
        # construct the pool1
        pool1 = tf.nn.max_pool2d(conv1, self.pool1_ksize, strides=self.pool1_strides, padding='VALID')
        pool1 = tf.nn.relu(pool1)
        # layer 3
        # construct the conv2
        conv2_w = self.init_weight(self.conv2_filter)
        conv2_b = self.init_bias(self.conv2_out_channels)
        conv2 = tf.nn.conv2d(pool1, conv2_w, strides=self.conv2_strides, padding='VALID') + conv2_b
        # layer 4
        # construct the pool2
        pool2 = tf.nn.max_pool2d(conv2, self.pool2_ksize, strides=self.pool2_strides, padding='VALID')
        pool2 = tf.nn.relu(pool2)
        # layer 5
        # construct the fc1
        fc1_w = self.init_weight([self.fc1_in_channels, self.fc1_out_channels])
        fc1_b = self.init_bias(self.fc1_out_channels)
        fc1 = tf.add(tf.matmul(tf.reshape(pool2, [-1, self.fc1_in_channels]), fc1_w), fc1_b)
        fc1 = tf.nn.relu(fc1)
        # layer 6
        # construct the fc2
        fc2_w = self.init_weight([self.fc1_out_channels, self.fc2_out_channels])
        fc2_b = self.init_bias(self.fc2_out_channels)
        fc2 = tf.add(tf.matmul(fc1, fc2_w), fc2_b)
        fc2 = tf.nn.relu(fc2)
        # layer 7
        # construct the final output
        out_w = self.init_weight([self.fc2_out_channels, self.final_out_channels])
        out_b = self.init_bias(self.final_out_channels)
        out = tf.add(tf.matmul(fc2, out_w), out_b)
        return tf.sigmoid(out)

    def forward(self, feats):
        """
        Forward the network
        """
        return self.net(feats)

    @staticmethod
    def init_weight(shape):
        """
        Init weight parameter.
        """
        w = tf.truncated_normal(shape=shape, mean=0, stddev=0.1)
        return tf.Variable(w)

    @staticmethod
    def init_bias(shape):
        """
        Init bias parameter.
        """
        b = tf.zeros(shape)
        return tf.Variable(b)
