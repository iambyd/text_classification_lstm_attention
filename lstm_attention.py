# -*- coding: utf-8 -*-
# @Author: snowbing
# @Date:   2019-07-21 17:33:07
# @Last Modified by:   Administrator
# @Last Modified time: 2019-07-21 17:35:43
# @Email: iambyd@live.cn
import tensorflow as tf


class RNN_Attention(object):
    """
    RNN+Attention模型
    """

    def __init__(self,
                 input_data,
                 output_data,
                 seq_len,
                 encode_config_file='./vocab_encode.pkl',
                 num_classes=2,
                 num_hidden=128,
                 embed_size=200,
                 batch_size=256,
                 epochs=10,
                 keep_prob_value=1.0,
                 lr=0.001):
        """
        ---------------定义模型参数-----------
        """
        # 指定训练输入数据集(dtype:Series)
        self.input_data = input_data
        # 指定训练预测数据集(dtype:Series)
        self.output_data = output_data
        # 指定句子固定长度
        self.seq_len = seq_len
        # 词语映射成的词向量长度
        self.embed_size = embed_size
        # 隐含层单元个数
        self.num_hidden = num_hidden
        # 结果分类个数
        self.num_classes = num_classes
        # batch大小
        self.batch_size = batch_size
        # 整个样本迭代次数
        self.epochs = epochs
        # dropout层概率
        self.keep_prob_value = keep_prob_value
        # 学习率
        self.lr = lr
        # 词汇量
        self.encode_config_file = encode_config_file
        with open(encode_config_file, 'rb') as f:
            vocab_encode = pickle.load(f)
        # 计算词语映射为词向量的个数，加1表示填充词
        self.vocab_num = len(vocab_encode) + 1
        del vocab_encode

        # ---------------模型输入-----------#
        self.x = tf.placeholder(
            shape=[None, self.seq_len],
            name='inputs',
            dtype=tf.int32
        )
        self.y = tf.placeholder(shape=[None, self.num_classes],
                                name='outputs',
                                dtype=tf.float32
                                )
        self.keep_prob = tf.placeholder(
            name='keep_prob',
            dtype=tf.float32
        )
        # ---------------添加embeding层-----------#
        # 将词向量化
        with tf.device('cpu:0'):
            with tf.name_scope('word2vec'):
                # 首先定义unk和pad的映射词向量为零向量
                pad_unk = tf.zeros(shape=(2, self.embed_size),
                                   dtype=tf.float32
                                   )
                self.embedding_train = tf.Variable(tf.truncated_normal(shape=(self.vocab_num - 2,
                                                                              self.embed_size
                                                                              ),
                                                                       mean=0.0,
                                                                       stddev=0.01
                                                                       ),
                                                   dtype=tf.float32
                                                   )
                self.embedding = tf.concat(
                    [pad_unk, self.embedding_train],
                    axis=0
                )
                tf.summary.histogram('embedding_train', self.embedding_train)
                # 将one-hot矩阵与词向量矩阵进行拼接，映射为指定长度的词向量
                self.embedding_inputs = tf.nn.embedding_lookup(
                    self.embedding,
                    self.x
                )
        # ---------------添加rnn层------------------#
        # 定义LSTM 单元
        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            self.num_hidden,
            forget_bias=0.1,
            state_is_tuple=True
        )
        with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):
            self.outputs, _ = tf.nn.dynamic_rnn(
                cell=self.lstm_cell,
                inputs=self.embedding_inputs,
                dtype=tf.float32
            )
        # ---------------添加attention层------------------#
        with tf.name_scope('Attention'):
            # 计算权重
            w_omega = tf.Variable(tf.random_normal(
                [self.num_hidden], stddev=0.1)
            )
            tf.summary.histogram('attention_w', w_omega)
        M = tf.tanh(self.outputs)
        self.alpha = tf.nn.softmax(
            tf.reshape(
                tf.matmul(
                    tf.reshape(M, [-1, self.num_hidden]),
                    tf.reshape(w_omega, [-1, 1])
                ),
                (-1, self.seq_len)
            )
        )
        tf.add_to_collection('alpha', self.alpha)
        r = tf.matmul(
            tf.transpose(self.outputs, [0, 2, 1]),
            tf.reshape(self.alpha, [-1, self.seq_len, 1])
        )
        r = tf.squeeze(r)
        # 将每个样本进行汇总成一个语义向量
        h_star = tf.tanh(r)
        # ---------------添加dropout层------------------#
        h_drop = tf.nn.dropout(h_star, self.keep_prob)
        # ---------------添加一个全连接层------------------#
        with tf.name_scope('Full_connected'):
            self.FC_w = tf.Variable(tf.random_normal(
                shape=[self.num_hidden, self.num_classes]),
                name='weights'
            )
            self.FC_b = tf.Variable(tf.random_normal(
                shape=[self.num_classes, ],
                name='biases')
            )
            self.results = tf.nn.xw_plus_b(h_drop, self.FC_w, self.FC_b)
        # 加上一个softmax函数将结果映射到[0,1]上,并保证每行和为1,得到各类别预测概率
        self.pred = tf.nn.softmax(self.results, name='pred')
        tf.add_to_collection('pred', self.pred)
        # ---------------模型评价------------------#
        with tf.name_scope('evaluate'):
            # 正确率
            self.accuracy = tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.argmax(self.pred, axis=1),
                             tf.argmax(self.y, axis=1)),
                    tf.float32
                )
            )
            # 交叉熵
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=self.results,
                    labels=self.y
                )
            )
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.scalar('cross_entropy', self.cross_entropy)
        # ---------------模型迭代器------------------#
        with tf.variable_scope('adam', reuse=tf.AUTO_REUSE):
            self.train_op = tf.train.AdamOptimizer(
                learning_rate=self.lr
            ).minimize(
                - self.cross_entropy
            )
