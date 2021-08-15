import sys, os
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from util.math_function import scaled_dot_product_attention, positional_encoding
import math

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
        self.dense_gelu = tf.keras.layers.Dense(d_ff, activation='gelu', use_bias = False)  # (batch_size, seq_len, dff)
        self.linear = tf.keras.layers.Dense(d_ff, use_bias = False)
        self.multiply = tf.keras.layers.Multiply()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.fin_dense = tf.keras.layers.Dense(d_model, use_bias = False)
        
    def call(self, hidden_states):
        hidden_gelu= self.dense_gelu(hidden_states)
        hidden_linear = self.linear(hidden_states)
        hidden_states = self.multiply([hidden_gelu, hidden_linear])
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fin_dense(hidden_states)
        
        return hidden_states

def dense_relu_dense(d_model, d_ff, dropout = 0.1): 
    return tf.keras.Sequential([
        tf.keras.layers.Dense(d_ff, activation='relu', use_bias = False),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dropout(dropout), # (batch_size, seq_len, d_model)
        tf.keras.layers.Dense(d_model, use_bias = False)  # (batch_size, seq_len, d_model)
    ])


class FeedForwardLayer(tf.keras.layers):
    def __init__(self, d_model, d_ff, feed_forward_proj = "gated-gelu", dropout=0.1):
        super(FeedForwardLayer, self).__init__()
        if feed_forward_proj == "relu":
            self.DenseReluDense =  dense_relu_dense(d_model, d_ff, dropout)
        else:
            self.DenseReluDense = Dense_GatedGelu_Dense(d_model, d_ff, dropout)

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states

def find_pruneable_heads_and_indices(
    heads, n_heads, head_size, already_pruned_heads
):
    """
    Finds the heads and their indices taking :obj:`already_pruned_heads` into account.
    Args:
        heads (:obj:`List[int]`): List of the indices of heads to prune.
        n_heads (:obj:`int`): The number of heads in the model.
        head_size (:obj:`int`): The size of each head.
        already_pruned_heads (:obj:`Set[int]`): A set of already pruned heads.
    Returns:
        :obj:`Tuple[Set[int], dtype = tf.int64]`: A tuple with the remaining heads and their corresponding indices.
    """
    mask = tf.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = tf.cast(tf.range(len(mask))[mask],dtype=tf.int64)
    return heads, index

def tf_index_select(input_, dim, indices):
    """
    input_(tensor): input tensor
    dim(int): dimension
    indices(list): selected indices list
    """
    shape = input_.get_shape().as_list()
    if dim == -1:
        dim = len(shape)-1
    shape[dim] = 1

    tmp = []
    for idx in indices:
        begin = [0]*len(shape)
        begin[dim] = idx
        tmp.append(tf.slice(input_, begin, shape))
    res = tf.concat(tmp, axis=dim)

    return res

def prune_linear_layer(layer, index, dim = 0):
    """
    Prune a linear layer to keep only entries in index.
    Used to remove heads.
    Args:
        layer (:obj:`tf.keras.layers.Dense`): The layer to prune.
        index (:obj:`tf.int64`): The indices to keep in the layer.
        dim (:obj:`int`, `optional`, defaults to 0): The dimension on which to keep the indices.
    Returns:
        :obj:`tf.keras.layers.Dense`: The pruned layer as a new layer with :obj:`requires_grad=True`.
    """
    W = layer.get_weights()[0]
    W = tf_index_select(W, dim, index)
    is_bias = len(layer.get_weights()) > 1
    if is_bias is True:
        if dim == 1:
            b = layer.get_weights()[1]
        else:
            b = layer.get_weights()[1][index]
    new_size = list(tf.shape(layer.get_weights()[0]))
    new_size[dim] = len(index)
    new_layer = tf.keras.layers.Dense(new_size[0], use_ibas = is_bias)
    new_layer.setweights(layer.get_weights())
    """new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True"""
    return new_layer


class AttentionLayer(tf.keras.layers):
    def __init__(self, d_model, d_kv, num_heads, is_decoder, relative_attention_num_buckets, has_relative_attention_bias=False, dropout = 0.1):
        super(AttentionLayer, self).__init__()
        self.is_decoder = is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.d_model = d_model
        self.key_value_proj_dim = d_kv
        self.n_heads = num_heads
        self.dropout = dropout
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        self.q = tf.keras.layers.Dense(self.inner_dim, use_bias=False)
        self.k = tf.keras.layers.Dense(self.inner_dim, use_bias=False)
        self.v = tf.keras.layers.Dense(self.inner_dim, use_bias=False)
        self.o = tf.keras.layers.Dense(self.d_model, use_bias=False)

        if self.has_relative_attention_bias : 
            self.relative_attention_bias = tf.keras.layers.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )

        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += tf.cast(relative_position > 0,tf.int64) * num_buckets
            relative_position = tf.math.abs(relative_position)
        else:
            relative_position = -tf.math.minimum(relative_position, tf.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + tf.cast(
            tf.math.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact), tf.int64)
        relative_postion_if_large = tf.math.minimum(
            relative_postion_if_large, (num_buckets - 1) * tf.ones_like(relative_postion_if_large)
        )

        relative_buckets += tf.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

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
        attn1, attn_weights_block1 = self.mha1(normed_x, normed_x, normed_x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        out1 = x + self.dropout1(attn1, training=training)

        normed_out1 = self.layernorm2(out1)
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, normed_out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = attn2 + out1  # (batch_size, target_seq_len, d_model)

        normed_out2 = self.layernorm3(out2)
        ffn_output = self.ffn(normed_out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = ffn_output + out2  # (batch_size, target_seq_len, d_model)

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

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        
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
        enc_output = self.tokenizer(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        
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