import tensorflow as tf
import numpy as np


def scaled_dot_product_attention(q, k, v, mask):
    """
    计算注意力权重。
    要求：q,k,v，前面维度需要保持一样。
        k,v必须有匹配的倒数第二个维度，即：seq_len_k=seq_len_v
    :param q: shape(...,seq_len_q,depth)
    :param k: shape(...,seq_len_k,depth)
    :param v: shape(...,seq_len_v,depth_v)
    :param mask:  shape(...,seq_len_q,depth),默认为None
    :return: 注意力权重(seq_len_q,seq_len_k), 输出（...,seq_len_q,depth_v）
    """
    multul_qk = tf.matmul(q, k, transpose_b=True)
    # 缩放 matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = multul_qk / tf.math.sqrt(dk)

    # 将mask加入到缩放的张量上
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights


def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(
        q, k, v, None)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)


"""
多头注意力
1. 线性层并拆分成多头
2. 每一个头执行按比缩放的点积注意力，scaled_dot_product_attention
3. 多头拼接一起
4. 最后过一层线性层，降低维度

说明：
1. 多头机制是先拆分成多份，再拼接回来。而不是输入维度不变，执行多次scaled_dot_product_attention，最后再缩放到原本维度。
2. 分拆后，每个头部的维度减少，因此总的计算成本与有全部维度的单头计算相同。
3. 多头只需拆最后一位维！
"""


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        分拆最后一个维度, d_model -> (num_heads, depth),
        :param x: (batch_size, seq_len, d_model)
        :param batch_size:
        :return: 转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))  # shape(batch_size, seq_len, num_head, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        """
        执行多头机制
        :param v: (batch_size, seq_len_q, d_model)
        :param k:
        :param q:
        :param mask:
        :return:
        output： (batch_size, seq_len_q, d_model)
        attention_weights: (batch_size, num_heads, seq_len_q, depth)
        """
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # 拆分成多头，其中d_model = num_heads * depth
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_q, depth)

        # 多个头的QKV共同执行scaled_dot_product_attention
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

        # 个头的注意力输出连接起来（用tf.transpose 和 tf.reshape）
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        # concat_attention: (batch_size, seq_len_q, d_model)

        # 放入最后的 Dense 层
        output = self.dense(concat_attention)
        return output, attention_weights


# temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
# y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
# out, attn = temp_mha(y, k=y, q=y, mask=None)
# print(out.shape)
# print(attn.shape)

"""
位置编码,
因为该模型并不包括任何的循环（recurrence）或卷积，所以模型添加了位置编码，为模型提供一些关于单词在句子中相对位置的信息。
"""


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(max_position, d_model):
    """

    :param max_position: 最大位置
    :param d_model:
    :return:  shape(1, max_position, d_model)
    """
    angle_rads = get_angles(np.arange(max_position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # 将 sin 应用于数组中的偶数索引（indices）；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 将 cos 应用于数组中的奇数索引；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


"""
masking
遮挡序列中所有填充的位置。确保模型不会将填充作为输入。
填充位置为1，其他位置为0
"""


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # 添加额外的维度来将填充加到注意力对数（logits）
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


"""
解码的时候前瞻遮挡（look-ahead mask），
用于遮挡序列中的后续标记（future token）。
就是说，预测t位置的词，需要屏蔽t后面位置的token，只是用1-t-1位置的token。
实现方法：
构造一个对角线以及对角线下方都是0，上方都是1的mask矩阵。
"""


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # ((seq_len, seq_len))


"""
Point wise feed forward network,
点式前馈网络，两个全连接层，两层之间有一个ReLU激活函数
"""


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


"""
优化器,
实现自定义的学习速率调度程序（scheduler）配合使用

"""


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * self.warmup_steps ** -1.5
        return tf.math.sqrt(self.d_model) * tf.math.minimum(arg1, arg2)
