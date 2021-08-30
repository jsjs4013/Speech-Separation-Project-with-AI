import sys, os
from numpy.core.records import array
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

#from util.math_function import scaled_dot_product_attention, positional_encoding
#from util.modeling_function import *
from modelOutput import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqModelOutput, BaseModelOutput
import math

class Dense_GatedGelu_Dense(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, dropout = 0.1, factor=1.):
        super(Dense_GatedGelu_Dense, self).__init__()
        d_model_init = tf.keras.initializers.RandomNormal(mean=0, stddev= factor * (d_model**-0.5))
        d_ff_init = tf.keras.initializers.RandomNormal(mean=0, stddev= factor * (d_ff**-0.5))
        self.dense_gelu = tf.keras.layers.Dense(d_ff, activation='gelu', kernel_initializer = d_model_init, use_bias = False)  # (batch_size, seq_len, dff)
        self.linear = tf.keras.layers.Dense(d_ff, kernel_initializer = d_model_init, use_bias = False)
        self.multiply = tf.keras.layers.Multiply()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.fin_dense = tf.keras.layers.Dense(d_model, kernel_initializer = d_ff_init, use_bias = False)
        
    def call(self, hidden_states):
        hidden_gelu= self.dense_gelu(hidden_states)
        hidden_linear = self.linear(hidden_states)
        hidden_states = self.multiply([hidden_gelu, hidden_linear])
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fin_dense(hidden_states)
        
        return hidden_states

def dense_relu_dense(d_model, d_ff, dropout = 0.1, factor = 1.): 
    d_model_init = tf.keras.initializers.RandomNormal(mean=0, stddev= factor * (d_model**-0.5))
    d_ff_init = tf.keras.initializers.RandomNormal(mean=0, stddev= factor * (d_ff**-0.5))
    return tf.keras.Sequential([
        tf.keras.layers.Dense(d_ff, activation='relu', kernel_initializer = d_model_init, use_bias = False),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dropout(dropout), # (batch_size, seq_len, d_model)
        tf.keras.layers.Dense(d_model, kernel_initializer = d_ff_init, use_bias = False)  # (batch_size, seq_len, d_model)
    ])


class FeedForwardLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, feed_forward_proj = "gated-gelu", dropout=0.1, factor=1.):
        super(FeedForwardLayer, self).__init__()
        if feed_forward_proj == "relu":
            self.DenseReluDense =  dense_relu_dense(d_model, d_ff, dropout, factor)
        else:
            self.DenseReluDense = Dense_GatedGelu_Dense(d_model, d_ff, dropout, factor)

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
    mask = tf.ones([n_heads, head_size])
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = tf.eqaul(tf.reshape(mask, -1),1)
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
    new_layer = tf.keras.layers.Dense(new_size[0], use_bias = is_bias)
    new_layer.set_weights(layer.get_weights())
    """new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True"""
    return new_layer


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, d_kv, num_heads, is_decoder, relative_attention_num_buckets, has_relative_attention_bias=False, dropout = 0.1, factor=1.):
        super(AttentionLayer, self).__init__()

        q_initializer = tf.keras.initializers.RandomNormal(mean = 0., stddev=factor * ((d_model * d_kv) ** -0.5))
        kvb_initializer = tf.keras.initializers.RandomNormal(mean = 0., stddev=factor * ((d_model) ** -0.5))
        o_initializer = tf.keras.initializers.RandomNormal(mean = 0., stddev=factor * ((num_heads * d_kv) ** -0.5))
        self.is_decoder = is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.d_model = d_model
        self.key_value_proj_dim = d_kv
        self.n_heads = num_heads
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.softmax = tf.keras.layers.Softmax()

        self.q = tf.keras.layers.Dense(self.inner_dim, kernel_initializer= q_initializer, use_bias=False)
        self.k = tf.keras.layers.Dense(self.inner_dim, kernel_initializer= kvb_initializer, use_bias=False)
        self.v = tf.keras.layers.Dense(self.inner_dim, kernel_initializer= kvb_initializer, use_bias=False)
        self.o = tf.keras.layers.Dense(self.d_model, kernel_initializer= o_initializer, use_bias=False)

        if self.has_relative_attention_bias : 
            self.relative_attention_bias = tf.keras.layers.Embedding(self.relative_attention_num_buckets, self.n_heads, embeddings_initializer= o_initializer)
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
            tf.math.log(tf.cast(relative_position, tf.float32) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact), tf.int64)
        relative_postion_if_large = tf.math.minimum(
            relative_postion_if_large, (num_buckets - 1) * tf.ones_like(relative_postion_if_large)
        )

        relative_buckets += tf.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets
    
    def compute_bias(self, query_length, key_length): 
        """Compute binned relative position bias"""
        context_position = tf.range(query_length, dtype=tf.int64)[:, None]
        memory_position = tf.range(key_length, dtype=tf.int64)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
        )
        #relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = tf.expand_dims(tf.transpose(values, perm=[2, 0, 1]), axis=0)  # shape (1, num_heads, query_length, key_length)
        return values

    def call(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        #batch_size, seq_length = tf.shape(hidden_states)[:2]
        batch_size = tf.shape(hidden_states)[0]
        seq_length = tf.shape(hidden_states)[1]
        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += tf.shape(past_key_value[0])[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else tf.shape(key_value_states)[1]

        def shape(states):
            """projection"""
            return tf.transpose(tf.reshape(states, (batch_size, -1, self.n_heads, self.key_value_proj_dim)),perm=[0,2,1,3])

        def unshape(states):
            """reshape"""
            return tf.reshape(tf.transpose(states, perm=[0,2,1,3]),(batch_size, -1, self.inner_dim))

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = tf.concat([past_key_value, hidden_states], axis=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = tf.matmul(
            query_states, tf.transpose(key_states, perm=[0, 1, 3, 2])
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = tf.zeros(
                    (1, self.n_heads, real_seq_length, key_length), dtype=scores.dtype
                )
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -tf.shape(hidden_states)[1] :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        scores += position_bias
        attn_weights = self.softmax(tf.cast(scores, dtype=tf.float32))  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = self.dropout(attn_weights)
        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(tf.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

class T5LayerSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, d_kv, num_heads, is_decoder, relative_attention_num_buckets, eps = 1e-6, has_relative_attention_bias=False, dropout=0.1, factor=1.):
        super(T5LayerSelfAttention, self).__init__()
        self.SelfAttention = AttentionLayer(d_model, d_kv, num_heads, is_decoder, relative_attention_num_buckets, has_relative_attention_bias=has_relative_attention_bias, dropout=dropout, factor=factor)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=eps)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs

class T5LayerCrossAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, d_kv, num_heads, is_decoder, relative_attention_num_buckets, eps = 1e-6, dropout=0.1, factor=1.):
        super(T5LayerCrossAttention, self).__init__()
        self.EncDecAttention = AttentionLayer(d_model, d_kv, num_heads, is_decoder, relative_attention_num_buckets, has_relative_attention_bias=False, dropout=dropout, factor= factor)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=eps)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs

class T5Block(tf.keras.layers.Layer):
    def __init__(self,  d_model, d_ff, d_kv, feed_forward_proj, num_heads, is_decoder, relative_attention_num_buckets, eps = 1e-6,  has_relative_attention_bias=False, dropout=0.1, factor=1.):
        super(T5Block, self).__init__()
        self.is_decoder = is_decoder
        self.layer = tf.keras.Sequential()
        self.layer.add(T5LayerSelfAttention(d_model, d_kv, num_heads, is_decoder, relative_attention_num_buckets, eps = eps, has_relative_attention_bias=has_relative_attention_bias, dropout=dropout, factor=factor))
        if self.is_decoder:
            self.layer.add(T5LayerCrossAttention(d_model, d_kv, num_heads, is_decoder, relative_attention_num_buckets, eps = eps, dropout= dropout, factor=factor))
        self.layer.add(FeedForwardLayer(d_model, d_ff, feed_forward_proj, dropout, factor=factor))

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):

        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer.layers[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == tf.float16 and tf.reduce_any(tf.math.isinf(hidden_states)):
            clamp_value = tf.experimental.numpy.finfo(hidden_states.dtype).max - 1000
            hidden_states = tf.clip_by_value(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer.layers[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == tf.float16 and tf.reduce_any(tf.math.isinf(hidden_states)):
                clamp_value = tf.experimental.numpy.finfo(hidden_states.dtype).max - 1000
                hidden_states = tf.clip_by_value(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer.layers[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == tf.float16 and tf.reduce_any(tf.math.isinf(hidden_states)):
            clamp_value = tf.experimental.numpy.finfo(hidden_states.dtype).max - 1000
            hidden_states = tf.clip_by_value(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)

class T5Stack(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, d_ff, d_kv, feed_forward_proj, num_heads, is_decoder, relative_attention_num_buckets, eps = 1e-6, dropout=0.1, embed_tokens=None, factor=1.):
        super(T5Stack, self).__init__()

        self.embed_tokens = embed_tokens
        self.is_decoder = is_decoder

        self.block = tf.keras.Sequential(
            [T5Block(d_model, d_ff, d_kv, feed_forward_proj, num_heads, is_decoder, relative_attention_num_buckets, eps = eps, dropout=dropout, has_relative_attention_bias=bool(i == 0), factor=factor) for i in range(num_layers)]
        )
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=eps)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.num_layers= num_layers
        #self.init_weights()

    """
    def init_weights(self):

        If needed prunes and maybe initializes weights.

        # Prune heads if needed
        if self.pruned_heads:
            self.prune_heads(self.pruned_heads)

        if _init_weights:
            # Initialize weights
            self.apply(self._init_weights)

            # Tie weights should be skipped when not initializing all weights
            # since from_pretrained(...) calls tie weights anyways
            self.tie_weights()

    def tie_weights(self):

        Tie the weights between the input embeddings and the output embeddings.

        If the :obj:`torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning
        the weights instead.

        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None and self.config.tie_word_embeddings:
            self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

        if self.config.is_encoder_decoder and self.config.tie_encoder_decoder:
            if hasattr(self, self.base_model_prefix):
                self = getattr(self, self.base_model_prefix)
            self._tie_encoder_decoder_weights(self.encoder, self.decoder, self.base_model_prefix)

        for module in self.modules():
            if hasattr(module, "_tie_weights"):
                module._tie_weights()


    def _init_weights(self, module):

        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (T5Model, T5ForConditionalGeneration, T5EncoderModel)):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, T5DenseReluDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5DenseGatedGeluDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))"""

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def call(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else False
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        return_dict = return_dict if return_dict is not None else False

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = tf.shape(input_ids)
        #    input_ids = tf.reshape(input_ids, (-1, input_shape[-1]))
        elif inputs_embeds is not None:
            input_shape = tf.shape(inputs_embeds)[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size = input_shape[0]
        seq_length = input_shape[1]

        # required mask seq length can be calculated via length of past
        mask_seq_length = tf.shape(past_key_values[0][0])[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f":obj:`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None: # tf.device(tf.ones(batch_size, mask_seq_length), input_embeds.device)
            """raise ValueError(
                    "you need to set an attention mask for input_ids"
            )"""
            attention_mask = tf.ones([batch_size, mask_seq_length])
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = tf.shape(encoder_hidden_states)[1]
            encoder_attention_mask = tf.ones(
                [batch_size, encoder_seq_length], dtype=tf.int64
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block.layers)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        #extended_attention_mask = attention_mask

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size = tf.shape(encoder_hidden_states)[0]
            encoder_sequence_length = tf.shape(encoder_hidden_states)[1]
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = tf.ones(encoder_hidden_shape) # , device=inputs_embeds.device
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block.layers, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            """if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)"""
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            """if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))"""

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
    def get_extended_attention_mask(self, attention_mask, input_shape):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        """if attention_mask.get_shape().rank == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.get_shape().rank == 2:"""
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder:
            batch_size = input_shape[0]
            seq_length = input_shape[1]
            seq_ids = tf.range(seq_length) # , device=device
            causal_mask = tf.tile(seq_ids[None, None, :],[batch_size, seq_length, 1]) <= seq_ids[None, :, None]
            # in case past_key_values are used we need to add a prefix ones mask to the causal mask
            # causal and attention masks must have same type with pytorch version < 1.3
            causal_mask = tf.cast(causal_mask, attention_mask.dtype)

            if tf.shape(causal_mask)[1] < tf.shape(attention_mask)[1]:
                prefix_seq_len = tf.shape(attention_mask)[1] - tf.shape(causal_mask)[1]
                causal_mask = tf.concat(
                    [
                        tf.ones(
                            [batch_size, seq_length, prefix_seq_len], dtype=causal_mask.dtype # device=device,
                        ),
                        causal_mask,
                    ],
                    axis=-1,
                )

            extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
        """else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )
    """
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(extended_attention_mask,dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def invert_attention_mask(self, encoder_attention_mask):
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (:obj:`torch.Tensor`): An attention mask.

        Returns:
            :obj:`torch.Tensor`: The inverted attention mask.
        """
        """if encoder_attention_mask.get_shape().rank == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]"""
        #if encoder_attention_mask.get_shape().rank == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = tf.cast(encoder_extended_attention_mask,dtype=self.dtype)  # fp16 compatibility

        if self.dtype == tf.float16:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e4
        elif self.dtype == tf.float32:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        else:
            raise ValueError(
                f"{self.dtype} not recognized. `dtype` should be set to either `torch.float32` or `torch.float16`"
            )

        return encoder_extended_attention_mask

    def get_head_mask(
        self, head_mask, num_hidden_layers: int, is_attention_chunked: bool = False
    ):
        """
        Prepare the head mask if needed.

        Args:
            head_mask (:obj:`torch.Tensor` with shape :obj:`[num_heads]` or :obj:`[num_hidden_layers x num_heads]`, `optional`):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (:obj:`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            :obj:`torch.Tensor` with shape :obj:`[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or
            list with :obj:`[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = _convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = tf.expand_dims(head_mask,axis=-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        #if head_mask.get_shape().rank == 1:
        mask_size = head_mask.get_shape()[0]
        head_mask = tf.expand_dims(head_mask, axis=0)
        head_mask = tf.expand_dims(head_mask, axis=0)
        head_mask = tf.expand_dims(head_mask, axis=-1)
        head_mask = tf.expand_dims(head_mask, axis=-1)
        head_mask = tf.broadcast_to(head_mask, [num_hidden_layers, 1, mask_size, 1, 1])
        """elif head_mask.get_shape().dims == 2:
            head_mask = tf.expand_dims(head_mask, axis=1)
            head_mask = tf.expand_dims(head_mask, axis=-1)
            head_mask = tf.expand_dims(head_mask, axis=-1)"""
        #assert head_mask.get_shape().rank == 5, f"head_mask.dim != 5, instead {len(head_mask.get_shape())}"
        head_mask = tf.cast(head_mask, dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask
class T5Model(tf.keras.Model):
    def __init__(self, vocab_size, num_layers, d_model, d_ff, d_kv, feed_forward_proj, num_heads, relative_attention_num_buckets, eps = 1e-6, dropout=0.1, embed_tokens=None, factor=1., embed_or_dense="embed"):
        super(T5Model, self).__init__()
        default_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=factor*1.)
        if embed_or_dense == "embed":
            self.shared = tf.keras.layers.Embedding(vocab_size, d_model, embeddings_initializer=default_initializer)
        else:
            self.shared = tf.keras.layers.Dense(d_model, kernel_initializer=default_initializer, use_bias=False, activation = 'tanh')

        self.encoder = T5Stack(num_layers, d_model, d_ff, d_kv, feed_forward_proj, num_heads, is_decoder = False, relative_attention_num_buckets=relative_attention_num_buckets, eps = eps, dropout=dropout, embed_tokens = self.shared, factor=factor)

        self.decoder = T5Stack(num_layers, d_model, d_ff, d_kv, feed_forward_proj, num_heads, is_decoder = True, relative_attention_num_buckets=relative_attention_num_buckets, eps = eps, dropout=dropout, embed_tokens = self.shared, factor=factor)
        self.num_layers = num_layers
        
        #self.init_weights()

        # Model parallel
        #self.model_parallel = False
        #self.device_map = None

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def call( # 결국 들어오는 것 : input_ids, attention_mask, decoder_input_ids, labels 
        self,
        input_ids=None, #
        attention_mask=None, #
        decoder_input_ids=None, #
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:

        Example::

            >>> from transformers import T5Tokenizer, T5Model

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5Model.from_pretrained('t5-small')

            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

            >>> last_hidden_states = outputs.last_hidden_state
        """
        
        use_cache = use_cache if use_cache is not None else False
        return_dict = return_dict if return_dict is not None else False

        # Encode if needed (training, first prediction pass)
        
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        """if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)"""

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class T5ModelMaskCreationModel(tf.keras.Model):
    def __init__(self, vocab_size, num_layers, d_model, d_ff, d_kv, feed_forward_proj, 
        num_heads, relative_attention_num_buckets, eps = 1e-6, dropout=0.1, embed_tokens=None, 
        factor=1., embed_or_dense="embed"):
        super(T5ModelMaskCreationModel, self).__init__()
        self.t5 = T5Model(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff, d_kv = d_kv, vocab_size=0, feed_forward_proj = feed_forward_proj, 
            relative_attention_num_buckets=relative_attention_num_buckets, eps=eps, dropout=dropout, factor=factor,
            embed_or_dense=embed_or_dense)
            
        self.mtp = tf.keras.layers.Multiply()
        
        self.concat = tf.keras.layers.Concatenate()

    def call( # 결국 들어오는 것 : input_ids, attention_mask, decoder_input_ids, labels 
        self,
        input_ids=None, #
        attention_mask=None, #
        decoder_input_ids=None, #
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, 
             training=False) # (batch_size, tar_seq_len, target_vocab_size)
        #inputs = tf.concat([inp,inp],2)
        #print('mask_output',final_output.shape)
        
        #final_output = self.mtp([final_output[:,:-1,:], inputs])
        # added Layers for Speech Separation
        
        # Multiply with Mask creatd by Transformers
        inputs = self.concat([input_ids[:,:decoder_input_ids.shape[1],:], input_ids[:,:decoder_input_ids.shape[1],:]])
        
        final_output = self.mtp([outputs[0], inputs])
        """
        final_output = self.dropout(final_output, training = training)
        pred1 = self.maskLayer1(final_output)
        pred2 = self.maskLayer2(final_output)
        
        cleaned1 = Multiply()([pred1, inp])
        cleaned2 = Multiply()([pred2, inp])
        
        final_output = Concatenate()([cleaned1, cleaned2])
        """
        return final_output