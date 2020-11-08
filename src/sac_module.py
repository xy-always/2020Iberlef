import tensorflow as tf
import numpy as np
import math

class Cnn_extractor(object):
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.sw0 = tf.layers.Conv1D(self.hidden_dim, 1, padding='same')
        self.bn0 = tf.layers.BatchNormalization()
        self.sw1 = tf.layers.Conv1D(self.hidden_dim, 1, padding='same')
        self.bn1 = tf.layers.BatchNormalization()
        self.sw2 = tf.layers.Conv1D(self.hidden_dim, 3, padding='same')
        self.bn2 = tf.layers.BatchNormalization()
        self.sw2_2 = tf.layers.Conv1D(self.hidden_dim, 3, padding='same')
        self.bn2_2 = tf.layers.BatchNormalization()
        self.sw3 = tf.layers.Conv1D(self.hidden_dim, 5, padding='same')
        self.bn3 = tf.layers.BatchNormalization()
        self.sw3_2 = tf.layers.Conv1D(self.hidden_dim, 5, padding='same')
        self.bn3_2 = tf.layers.BatchNormalization()
        self.sw3_3 = tf.layers.Conv1D(self.hidden_dim, 5, padding='same')
        self.bn3_3 = tf.layers.BatchNormalization()

    def __call__(self, input):
        with tf.variable_scope('cnn_extractor'):
            input = self.sw0(input)
            input = tf.nn.selu(input)
            input = self.bn0(input)
            sw1 = self.sw1(input)
            sw1 = tf.nn.selu(sw1)
            sw1 = self.bn1(sw1)
            sw2 = self.sw2(input)
            sw2 = tf.nn.selu(sw2)
            sw2 = self.bn2(sw2)
            sw2 = self.sw2_2(sw2)
            sw2 = tf.nn.selu(sw2)
            sw2 = self.bn2_2(sw2)
            sw3 = self.sw3(input)
            sw3 = tf.nn.selu(sw3)
            sw3 = self.bn3(sw3)
            sw3 = self.sw3_2(sw3)
            sw3 = tf.nn.selu(sw3)
            sw3 = self.bn3_2(sw3)
            sw3 = self.sw3_3(sw3)
            sw3 = tf.nn.selu(sw3)
            sw3 = self.bn3_3(sw3)
            
            cnn_output = tf.concat([sw1, sw2, sw3], -1)
            cnn_output = tf.layers.dense(cnn_output, self.hidden_dim, activation=tf.nn.selu)
            return tf.nn.dropout(cnn_output, keep_prob=0.5)


class Attention(object):
    def __init__(self, hidden_dim, num_tags):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_tags = num_tags

        self.attn_dense = tf.layers.Dense(self.hidden_dim, use_bias=False,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.attn_linear = tf.layers.Dense(self.hidden_dim, use_bias=True, activation=tf.nn.tanh,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           bias_initializer=tf.zeros_initializer())
        self.__init_embs()

    def __init_embs(self):
        with tf.variable_scope('tag_embedding'):
            self._tag_embeddings = tf.get_variable(name='_tag_embeddings', shape=[self.num_tags, 25], dtype=tf.float32)


    def __call__(self, input, sequence_lengths):
        with tf.variable_scope('attention'):
            tag_embeddings = tf.nn.embedding_lookup(params=self._tag_embeddings,
                                                         ids=tf.constant(list(range(self.num_tags)), dtype=tf.int32),
                                                         name='tag_embeddings')
            print('input shape:', input.shape)
            query = tf.transpose(input, [1, 0, 2])
            max_time = tf.shape(query)[0]
            batch_size = tf.shape(query)[1]
            context = tf.tile(tf.expand_dims(tag_embeddings, 0),
                              [batch_size, 1, 1])
            print('context shape:', context.shape)
            query_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
            query_ta = query_ta.unstack(query)
            attn_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
            output_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
            t0 = tf.constant(0, dtype=tf.int32, name='t0')
            f0 = tf.zeros([batch_size], dtype=tf.bool, name='f0')
            print('t0 shape:', t0.shape)
            # print('attn_ta shape:', attn_ta.shape)
            # print('output_ta shape:', output_ta.shape)
            print('f0 shape:', f0.shape)

            def loop_fn(t, attn_ta, output_ta, f):
                cur_q = query_ta.read(t)
                gamma_h = self.attn_dense(context)
                gamma_h = tf.squeeze(tf.matmul(gamma_h, tf.expand_dims(cur_q, -1)), -1)
                weights = tf.nn.softmax(gamma_h, -1)
                c_t = tf.squeeze(tf.matmul(tf.expand_dims(weights, 1), context), 1)
                output = self.attn_linear(tf.concat([c_t, cur_q], -1))
                attn_ta = attn_ta.write(t, gamma_h)
                output_ta = output_ta.write(t, output)
                f = tf.greater_equal(t + 1, sequence_lengths)
                return t + 1, attn_ta, output_ta, f

            _, attn_ta, output_ta, _ = tf.while_loop(
                
                cond=lambda _1, _2, _3, f: tf.logical_not(tf.reduce_all(f)),
                body=loop_fn,
                loop_vars=(t0, attn_ta, output_ta, f0)
            )
            
            self.attn_cnn_outputs = tf.transpose(output_ta.stack(), [1, 0, 2])
            attn_weights = tf.transpose(attn_ta.stack(), [1, 0, 2])
            return attn_weights, self.attn_cnn_outputs

class Position_Encoder(object):
    def __init__(self, emb_size, max_len=5000):
        self.emb_size = emb_size
        self.max_len = max_len
        pe = np.zeros([max_len, emb_size], np.float32)
        position = np.expand_dims(np.arange(0, max_len), 1).astype(np.float32)
        div_term = np.exp(np.arange(0 ,emb_size, 2).astype(np.float32) * -(math.log(10000.0) / emb_size))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = np.expand_dims(pe, 1)
        self.pe = tf.Variable(pe, trainable=False)

    def __call__(self, inputs, seq_length):
        with tf.variable_scope('position_encoder'):
            embs = tf.transpose(inputs, [1, 0, 2])
            max_time = tf.shape(embs)[0]
            batch_size = tf.shape(embs)[1]
            embs = embs * tf.sqrt(float(self.emb_size))
            embs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
            embs_ta = embs_ta.unstack(embs)
            output_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
            t0 = tf.constant(0, dtype=tf.int32)
            f0 = tf.zeros([batch_size], dtype=tf.bool)
            mask = tf.expand_dims(tf.cast(tf.sequence_mask(seq_length), tf.float32), -1)
            def loop_fn(t, output_ta, f):
                cur_emb = embs_ta.read(t)
                output = tf.concat([cur_emb, tf.tile(self.pe[t], [batch_size, 1])], -1)
                output_ta = output_ta.write(t, output)
                f = tf.greater_equal(t + 1, seq_length)
                return t + 1, output_ta, f

            _, output_ta, _ = tf.while_loop(
                cond=lambda _1, _2, f: tf.logical_not(tf.reduce_all(f)),
                body=loop_fn,
                loop_vars=(t0, output_ta, f0)
            )
            embs = tf.transpose(output_ta.stack(), [1, 0, 2])
            embs *= mask
            return embs
