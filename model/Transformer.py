"""
解码器-编码器
1. 输入语句经过n层编码器，为序列中的每个词/标记成一个输出向量。
2. 解码器根据解码器的输出以及他自身的输入（自注意力）来预测下一个词。

"""

import tensorflow as tf

from model.functions import *

"""
编码器层：
每个编码层包括：
1. 多头注意力（有填充遮挡）
2. 点式前馈网络（Point wise feed forward networks）。
每个子层在其周围有一个残差链接，然后进行归一化。残差有助于避免深度网络中的梯度消失问题。
每个子层的输出是 LayerNorm(x + Sublayer(x))
归一化是在 d_model（最后一个）维度完成的。
Transformer 中有 N 个编码器层。
"""


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training)
        # 加入残差和layernorm
        out1 = self.layernorm1(attn_output + x)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2


"""
解码器层
1. 遮挡的多头注意力（前瞻遮挡和填充遮挡）
2. 多头注意力（填充遮挡）。V（数值）和k（主键 ）接收编码器输出作为输入。Q（请求）接受遮挡的多头注意力子层作为输出。
3. 点式前馈网络
"""


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        :param x:
        :param enc_output: enc_output.shape == (batch_size, input_seq_len, d_model)
        :param training:
        :param look_ahead_mask:
        :param padding_mask:
        :return:
        """

        # 编码器输入,这一层使用look_ahead_mask，
        attn_output1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn_output1 = self.dropout1(attn_output1, training)
        out1 = self.layernorm1(attn_output1 + x)

        # 遮挡的解码器输出，这一层使用padding_mask，因为这里本质还是使用encoder输入序列，key=out1
        attn_output2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1,
                                                      padding_mask)  # (batch_size, target_seq_len, d_model)
        attn_output2 = self.dropout2(attn_output2, training)
        out2 = self.layernorm2(attn_output2 + out1)  # (batch_size, target_seq_len, d_model), 用key做norm

        # feed forward层
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training)
        out3 = self.layernorm3(ffn_output + out2)
        return out3, attn_weights_block1, attn_weights_block2


"""
编码器（Encoder）
1. input embedding
2. positional encoding
3. n个编码器层
"""


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)  # shape=(1, maximum_position_encoding, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # input_embedding与positional_encoding编码相加，
        # result = input_embedding * sqrt(d_model) + positional_encoding
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # 取前seq_len位置的位置embedding
        x_pos_encoding = self.pos_encoding[:, :seq_len, :]  # (1, input_seq_len, d_model)
        x += x_pos_encoding  # (batch_size, input_seq_len, d_model)
        # 构造输入之后加一层dropout
        x = self.dropout(x, training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x  # (batch_size, input_seq_len, d_model)


# sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
#                          dff=2048, input_vocab_size=8500,
#                          maximum_position_encoding=10000)
#
# sample_encoder_output = sample_encoder(tf.random.uniform((64, 62)),
#                                        training=False, mask=None)
#
# print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)


"""
解码器：
1. 输出嵌入（Output Embedding）
2. 位置编码（Positional Encoding）
3. N 个解码器层（decoder layers）
目标（target）经过一个嵌入后，改嵌入与位置编码相加，结果作为解码器层的输入。
"""


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        # 输入后一层就要加一个droupout
        x = self.dropout(x, training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return x, attention_weights  # (batch_size, target_seq_len, d_model)


# sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
#                          dff=2048, target_vocab_size=8000,
#                          maximum_position_encoding=5000)
#
# output, attn = sample_decoder(tf.random.uniform((64, 26)),
#                               enc_output=sample_encoder_output,
#                               training=False, look_ahead_mask=None,
#                               padding_mask=None)
#
# print(output.shape)
# print(attn['decoder_layer2_block2'].shape)

"""
构建Transformer
Transformer 包括编码器，解码器和最后的线性层。
解码器的输出是线性层的输入，返回线性层的输出
"""


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        # enc_output.shaape = (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, training, enc_padding_mask)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output, attention_weights


sample_transformer = Transformer(
    num_layers=2, d_model=512, num_heads=8, dff=2048,
    input_vocab_size=8500, target_vocab_size=8000,
    pe_input=10000, pe_target=6000)

temp_input = tf.random.uniform((64, 62))
temp_target = tf.random.uniform((64, 26))

fn_out, _ = sample_transformer(temp_input, temp_target, training=False,
                               enc_padding_mask=None,
                               look_ahead_mask=None,
                               dec_padding_mask=None)

print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)
