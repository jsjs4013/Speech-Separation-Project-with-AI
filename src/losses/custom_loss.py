import tensorflow as tf

mse_loss_object = tf.keras.losses.MeanSquaredError(
    reduction=tf.keras.losses.Reduction.NONE, name='mean_squared_error'
)
#tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def MSE_Custom_Loss(real, pred, length):
    # pred = (batch, seq_len, output_size)
    # mask (batch, seq_len)
    mask = tf.cast(tf.sequence_mask(tf.squeeze(length), tf.shape(pred)[1]), tf.float32) # tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = mse_loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def MSE_Custom_Loss_No_Length(real, pred):
    # pred = (batch, seq_len, output_size)
    # mask (batch, seq_len)

    padding = tf.zeros([1,tf.shape(pred)[-1]], tf.float32)
    not_equal_t = tf.equal(padding, real)
    mask = tf.math.logical_not(tf.math.reduce_all(not_equal_t, axis=2))
    
    loss_ = mse_loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

if __name__ == "__main__":
    real = tf.constant([
        [
            [
                1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9
            ],
            [
                1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9
            ],
            [
                1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9
            ],
        ],
        [
            [
                1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9
            ],
            [
                1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9
            ],
            [
                1,2,-1,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9
            ],
        ],
        [
            [
                1,0,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9
            ],
            [
                0,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9
            ],
            [
                1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,0
            ],
        ],
        [
            [
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
            ],
            [
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
            ],
            [
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
            ],
        ],
    ])
    real = tf.cast(real, tf.float32)
    pred = real
    print(MSE_Custom_Loss_No_Length(real, pred))