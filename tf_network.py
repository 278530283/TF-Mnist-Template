import tensorflow as tf


class LeNet:

    def __init__(self):
        """
        Define some basic parameters here
        """
        self.conv1_W = None
        self.conv1_b = None
        pool1 = None
        conv2_W = None
        conv2_b = None
        pool2 = None
        fc1_W = None
        fc1_b = None
        fc2_W = None
        fc2_b = None
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
        # TODO: construct the conv1
        # layer 2
        # TODO: construct the pool1
        # layer 3
        # TODO: construct the conv2
        # layer 4
        # TODO: construct the pool2
        # layer 5
        # TODO: construct the fc1
        # layer 2
        # TODO: construct the fc2

        # input = tf.Variable(tf.constant(1.0, shape=[1, 32, 32, 1]))
        conv1_w = self.init_weight([5, 5, 1, 6])
        conv1_b = self.init_bias(6)
        conv1 = tf.nn.conv2d(feats, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
        pool1 = tf.nn.avg_pool2d(conv1, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        pool1 = tf.nn.relu(pool1)
        conv2_w = self.init_weight([5, 5, 6, 16])
        conv2_b = self.init_bias(16)
        conv2 = tf.nn.conv2d(pool1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
        pool2 = tf.nn.avg_pool2d(conv2, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        pool2 = tf.nn.relu(pool2)
        fc1_w = self.init_weight([5 * 5 * 16, 120])
        fc1_b = self.init_bias(120)
        fc1 = tf.add(tf.matmul(tf.reshape(pool2, [-1, 5 * 5 * 16]), fc1_w), fc1_b)
        fc2_w = self.init_weight([120, 84])
        fc2_b = self.init_bias(84)
        fc2 = tf.add(tf.matmul(fc1, fc2_w), fc2_b)
        fc2 = tf.nn.relu(fc2)
        fc2 = tf.tile(fc2, [10, 1])
        out_w = self.init_weight([10, 84])
        logits = tf.reduce_sum(tf.square(tf.subtract(fc2, out_w)), 1)
        return logits

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
