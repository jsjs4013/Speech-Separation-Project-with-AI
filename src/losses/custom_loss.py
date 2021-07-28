import tensorflow as tf

loss_object = tf.keras.losses.MeanSquaredError(
    reduction=tf.keras.losses.Reduction.AUTO, name='mean_squared_error'
)
#tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def MSE_Custom_Loss(real, pred, length, MSE_loss_object):
    # pred = (batch, seq_len, output_size)
    # mask (batch, seq_len)
    mask = tf.cast(tf.sequence_mask(tf.squeeze(length), tf.shape(pred)[1]), tf.float32) # tf.math.logical_not(tf.math.equal(real, 0))
    print(mask)
    loss_ = MSE_loss_object(real, pred)
    print(loss_)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    print(loss_)

    return tf.reduce_sum(loss_)#/tf.reduce_sum(mask)