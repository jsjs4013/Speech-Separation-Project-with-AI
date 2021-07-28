import sys, os
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from util.math_function import scaled_dot_product_attention, positional_encoding

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads # 512/8 = 64

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

class Dense_GatedGelu_Dense(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super(Dense_GatedGelu_Dense, self).__init__()
        self.dense_gelu = tf.keras.layers.Dense(d_ff, activation='relu', use_bias = False)  # (batch_size, seq_len, dff)
        self.linear = tf.keras.layers.Dense(d_ff, use_bias = False)
        self.add_gates = tf.keras.layers.Add()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.fin_dense = tf.keras.layers.Dense(d_model, use_bias = False)
        
    def call(self, hidden_states):
        hidden_gelu= self.dense_gelu(hidden_states)
        hidden_linear = self.linear(hidden_states)
        hidden_states = self.add_gates([hidden_gelu, hidden_linear])
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fin_dense(hidden_states)
        
        return hidden_states

def dense_relu_dense(d_model, d_ff, dropout = 0.1): 
    return tf.keras.Sequential([
        tf.keras.layers.Dense(d_ff, activation='relu', use_bias = False),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dropout(dropout), # (batch_size, seq_len, d_model)
        tf.keras.layers.Dense(d_model, use_bias = False)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = Dense_GatedGelu_Dense(d_model, d_ff, dropout)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training, mask):

        normed_x = self.layernorm1(x)
        attn_output, _ = self.mha(normed_x, normed_x, normed_x, mask)  # (batch_size, input_seq_len, d_model)
        out1 = x + self.dropout1(attn_output, training=training)  # (batch_size, input_seq_len, d_model)

        normed_out1 = self.layernorm2(out1)
        ffn_output = self.ffn(normed_out1)  # (batch_size, input_seq_len, d_model)
        out2 = out1 + self.dropout2(ffn_output, training=training)  # (batch_size, input_seq_len, d_model)
        
        return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = Dense_GatedGelu_Dense(d_model, d_ff, dropout)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        normed_x = self.layernorm1(x)
        print('normed_x', normed_x.shape)
        attn1, attn_weights_block1 = self.mha1(normed_x, normed_x, normed_x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        print('attn1', attn1.shape)
        out1 = x + self.dropout1(attn1, training=training)

        normed_out1 = self.layernorm2(out1)
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, normed_out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        print('attn2', attn2.shape)
        attn2 = self.dropout2(attn2, training=training)
        out2 = attn2 + out1  # (batch_size, target_seq_len, d_model)

        normed_out2 = self.layernorm3(out2)
        ffn_output = self.ffn(normed_out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        print('ffn_output', ffn_output.shape)
        out3 = ffn_output + out2  # (batch_size, target_seq_len, d_model)
        print('out3', out3.shape)

        return out3, attn_weights_block1, attn_weights_block2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_size,
               maximum_position_encoding, dropout=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        #self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.embedding = tf.keras.layers.Dense(self.d_model, activation = 'tanh') # For STFT Input 
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, d_ff, dropout)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training, mask):
        print('x : ',x)

        seq_len = tf.shape(x)[1]
        print('seq_len : ',seq_len)

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        print('embedded : ',x.shape)
        
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, target_size, maximum_position_encoding, dropout=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        #self.embedding = tf.keras.layers.Embedding(target_size, d_model)
        self.embedding = tf.keras.layers.Dense(self.d_model, activation = 'tanh') # For STFT Input 
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, enc_output, training,look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                 look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_size,
               target_size, pe_input, pe_target, dropout=0.1):
        super(Transformer, self).__init__()

        self.tokenizer = Encoder(num_layers, d_model, num_heads, d_ff,
                                 input_size, pe_input, dropout)

        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff,
                               target_size, pe_target, dropout)

        self.final_layer = tf.keras.layers.Dense(target_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        # inp (4, 583, 129) -> *2 = (3, 583, 256)
        # tar (4, 584, 129)
        print('input : ',inp.shape)
        enc_output = self.tokenizer(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        print('enc_output',enc_output.shape)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        print('target : ',tar.shape)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        print('dec_output',enc_output.shape)
        
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

class TransformerSpeechSep(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_size,
               target_size, pe_input, pe_target, dropout=0.1):
        super(TransformerSpeechSep, self).__init__()

        self.transformer = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff,
            input_size=input_size, target_size=target_size,
            pe_input=pe_input, pe_target=pe_target, dropout=dropout)
            
        self.mtp = tf.keras.layers.Multiply()
        
        self.concat = tf.keras.layers.Concatenate()

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        
        final_output, attention_weights = self.transformer(inp, tar,training,enc_padding_mask,look_ahead_mask,dec_padding_mask)  # (batch_size, tar_seq_len, target_vocab_size)
        #inputs = tf.concat([inp,inp],2)
        #print('mask_output',final_output.shape)
        
        #final_output = self.mtp([final_output[:,:-1,:], inputs])
        print('final_output',final_output.shape)
        # added Layers for Speech Separation
        
        # Multiply with Mask creatd by Transformers
        inputs = self.concat([inp[:,:tar.shape[1],:], inp[:,:tar.shape[1],:]])
        
        final_output = self.mtp([final_output, inputs])
        """
        final_output = self.dropout(final_output, training = training)
        pred1 = self.maskLayer1(final_output)
        pred2 = self.maskLayer2(final_output)
        
        cleaned1 = Multiply()([pred1, inp])
        cleaned2 = Multiply()([pred2, inp])
        
        final_output = Concatenate()([cleaned1, cleaned2])
        """
        return final_output, attention_weights