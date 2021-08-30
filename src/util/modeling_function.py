import tensorflow as tf

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
        batch_size, seq_length = input_shape
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
                        (batch_size, seq_length, prefix_seq_len), dtype=causal_mask.dtype # device=device,
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