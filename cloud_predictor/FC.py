import tensorflow as tf


class FC:
    """Docs for FC"""

    def __init__(self, shapes=[4, 8, 32, 64, 256, 16, 4,  2], initializer=tf.contrib.layers.xavier_initializer(), path=None):
        """Constructor for FC"""

        self.initializer = initializer

        self.X_inp = tf.placeholder(dtype=tf.float32, shape=[None, shapes[0]])
        self.y_inp = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        self.keep_prob = tf.placeholder(dtype=tf.float32)

        #
        self.fc_inp = self.fc_initializer(self.X_inp, [shapes[0], shapes[1]], name='fc_inp', activation=tf.nn.relu)
        self.bn_inp = self.bn_initializer(self.fc_inp, shapes[1], 'bn_inp')
        self.fc_h1 = self.fc_initializer(self.bn_inp, [shapes[1], shapes[2]], name='fc_h1', activation=tf.nn.sigmoid)
        self.fc_h2 = self.fc_initializer(self.fc_h1, [shapes[2], shapes[3]], name='fc_h2', activation=tf.nn.sigmoid)
        self.drop_h2 = tf.nn.dropout(self.fc_h2, keep_prob=self.keep_prob)
        self.fc_h3 = self.fc_initializer(self.drop_h2, [shapes[3], shapes[4]], name='fc_h3', activation=tf.nn.sigmoid)
        self.fc_h4 = self.fc_initializer(self.fc_h3, [shapes[4], shapes[5]], name='fc_h4', activation=tf.nn.sigmoid)
        self.fc_h5 = self.fc_initializer(self.fc_h4, [shapes[5], shapes[6]], name='fc_h5', activation=tf.nn.sigmoid)
        self.y_logits = self.fc_initializer(self.fc_h5, [shapes[6], shapes[7]], name='fc_logits')

        self.y_predict = tf.nn.softmax(self.y_logits)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y_logits, labels=self.y_inp))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.arg_max(self.y_predict, 1), tf.arg_max(self.y_inp, 1)), tf.float32))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if path:
            tf.train.Saver().restore(self.sess, path)

    def train(self, batch_loader, n_epochs=2000000, batch_size=256, step_per_epoch=300):
        saver = tf.train.Saver()
        for epoch in range(n_epochs):
            for i in range(step_per_epoch):
                x_batch, y_batch = batch_loader.train.next_batch(batch_size)
                self.sess.run(self.optimizer, feed_dict={self.X_inp: x_batch, self.y_inp: y_batch, self.keep_prob: 0.5})
            acc_va, y_out, loss = self.sess.run([self.accuracy, self.y_predict, self.loss],
                                   feed_dict={self.X_inp: batch_loader.test.X, self.y_inp: batch_loader.test.y,
                                              self.keep_prob: 1})
            print('Epoch %d Step %d Accuracy %f Loss %f' % (epoch, i, acc_va, loss))
            print(y_out[:10], batch_loader.test.y[:10])
            saver.save(self.sess, save_path=r'models\model.ckpt', global_step=epoch)

    def predict(self, X):
        return self.sess.run(self.y_predict, feed_dict={self.X_inp: [X], self.keep_prob: 1})

    def fc_initializer(self, tensor, shape, name, activation=None):
        with tf.variable_scope(name):
            w = tf.get_variable(name=name + '_w', initializer=self.initializer, shape=shape)
            b = tf.get_variable(name=name + '_b', initializer=self.initializer, shape=[shape[1]])
            logits = tf.add(tf.matmul(tensor, w), b)

            if activation:
                return activation(logits)
            else:
                return logits

    def bn_initializer(self, tensor, size, name):
        mean, var = tf.nn.moments(tensor, [0])
        with tf.variable_scope(name):
            scale = tf.Variable(tf.zeros([size]))
            beta = tf.Variable(tf.zeros([size]))
            return tf.nn.batch_normalization(tensor, mean, var, beta, scale, 0.0001)

