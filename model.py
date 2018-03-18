# encoding=utf-8

from util import *


class model():
    def __init__(self, args, is_training, batch_size, num_steps, tagDic):
        # print "记录使用的batch大小和截断长度"
        self.batch_size = batch_size
        self.num_steps = num_steps
        # 是否训练
        self.is_training = is_training
        # 节点不被dropout的概率
        self.drop_pro = args.drop_pro
        # 深层循环神经网络中LSTM结构的层数
        self.num_layers = args.num_layers
        # 学习速率
        self.lrate = args.lrate
        # 隐藏层规模
        self.hidden_size = args.hidden_size
        # integer, the size of an attention window.
        self.attn_length = args.attn_length

        # print "定义输入层"  # 可以看到输入层的维度为batch_size * length * hidden_size，这和训练数据是一致的。
        self.input_data = tf.placeholder(tf.float32, shape=[None, self.num_steps, 200], name='input_data')

        # Print "定义样本真实长度"
        self.input_data_lengths = tf.placeholder(tf.int32, shape=[None], name='input_data_lengths')

        # print "定义预期输出"  # 它的维度应该和正确答案的维度也是一样的。
        self.targets = tf.placeholder(tf.int32, shape=[None, self.num_steps], name='targets')

        # "定义转移矩阵"
        self.trans_params = tf.placeholder(tf.float32, shape=[len(tagDic) + 1, len(tagDic) + 1], name='trans_params')

        # print "定义使用LSTM结构为循环体结构且使用dropout的深层循环神经网络"
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size)
        if self.is_training:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.drop_pro)

        lstm_cell = tf.contrib.rnn.AttentionCellWrapper(lstm_cell, self.attn_length, state_is_tuple=True)

        if self.is_training:
            self.input_data = tf.nn.dropout(self.input_data, self.drop_pro)

        self.outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell, sequence_length=self.input_data_lengths,
                                       inputs=self.input_data, dtype=tf.float32)

        self.weights = tf.get_variable("weights", [self.hidden_size, len(tagDic) + 1],
                                       initializer=tf.random_normal_initializer())
        self.biases = tf.get_variable("biases", [len(tagDic) + 1],
                                      initializer=tf.random_normal_initializer())
        matricized_outputs = tf.reshape(self.outputs, [-1, self.hidden_size])
        self.logits = tf.matmul(matricized_outputs, self.weights) + self.biases
        # print "将从LSTM中得到的输出再经过一个CRF后得到最后的预测结果"
        self.tags_scores = tf.reshape(self.logits, [self.batch_size, self.num_steps, len(tagDic) + 1])
        #
        self.sequence_lengths = tf.ones([self.batch_size], dtype=tf.int32) * (self.num_steps - 1)

        # 最终的预测结果经过CRF后在每一个时刻上都是一个数字。
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            inputs=self.tags_scores,
            # A [batch_size, max_seq_len, num_tags] tensor of unary potentials to use as input to the CRF layer.
            tag_indices=self.targets,
            # A [batch_size, max_seq_len] matrix of tag indices for which we compute the log-likelihood.
            sequence_lengths=self.sequence_lengths,
            # A [batch_size] vector of true sequence lengths.
            transition_params=self.trans_params
            #  A [num_tags, num_tags] transition matrix
        )
        self.loss = tf.reduce_mean(-log_likelihood)
        # print "最终状态"
        # self.final_state = states
        # 只在训练模型时定义反向传播操作。
        if not self.is_training:
            return
        trainable_variables = tf.trainable_variables()  # return a list of Variable objects.

        # 通过clip_by_global_norm函数控制梯度的大小，避免梯度膨胀的问题。
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, trainable_variables), 5
        )  # 5是用于控制梯度膨胀的参数
        learning_rate = self.lrate
        # print "定义优化方法"
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # 定义训练步骤
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
