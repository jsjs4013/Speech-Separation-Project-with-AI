import tensorflow as tf

mse_loss_object = tf.keras.losses.MeanSquaredError(
    reduction=tf.keras.losses.Reduction.NONE, name='mean_squared_error'
)
#tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def mse_with_proper_loss(output_size):
    def MSE_Custom_Loss(real, pred):
        # pred = (batch, seq_len, output_size)
        # mask (batch, seq_len)
        ori_length = tf.shape(real)[1]
        real_real = tf.slice(real, [0, 0, 0], [-1, ori_length-1, -1])
        lengths = tf.slice(real, [0, ori_length-1, 0], [-1, -1, 1]) 

        mask = tf.cast(tf.sequence_mask(tf.squeeze(lengths), tf.shape(pred)[1]), tf.float32) # tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = mse_loss_object(real_real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
    return MSE_Custom_Loss

# Custom PIT loss

def pit_with_outputsize(output_size):
    def pit_loss(y_true, y_pred):
        ori_length = tf.shape(y_true)[1]
        
        # Label & Length divide
        labels = tf.slice(y_true, [0, 0, 0], [-1, ori_length-1, -1]) # [batch_size, length_size, 129]
        lengths = tf.slice(y_true, [0, ori_length-1, 0], [-1, -1, 1]) # [batch_size, 1, 1]
        
        mask = tf.cast(tf.sequence_mask(tf.squeeze(lengths), tf.shape(y_pred)[1]), tf.float32)
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.tile(mask, [1, 1, output_size])

        # Label value slice
        labels1 = tf.slice(labels, [0, 0, 0], [-1, -1, output_size])
        labels2 = tf.slice(labels, [0, 0, output_size], [-1, -1, -1])

        # Predict value slice
        pred1 = tf.slice(y_pred, [0, 0, 0], [-1, -1, output_size])
        pred2 = tf.slice(y_pred, [0, 0, output_size], [-1, -1, -1])
        
        # Masking
        mask_pred1 = pred1 * mask
        mask_pred2 = pred2 * mask
        
        pred1_front = tf.slice(mask_pred1, [0, 0, 0], [-1, 1, output_size])
        pred2_front = tf.slice(mask_pred2, [0, 0, 0], [-1, 1, output_size])

        pred1_after = tf.slice(mask_pred1, [0, 1, 0], [-1, -1, output_size])
        pred2_after = tf.slice(mask_pred2, [0, 1, 0], [-1, -1, output_size])
        
        labels1_front = tf.slice(labels1, [0, 0, 0], [-1, 1, output_size])
        labels2_front = tf.slice(labels2, [0, 0, 0], [-1, 1, output_size])

        labels1_after = tf.slice(labels1, [0, 1, 0], [-1, -1, output_size])
        labels2_after = tf.slice(labels2, [0, 1, 0], [-1, -1, output_size])

        # Permute calculate (batch, seqlen, 258) mask = (batch, seq_len)
        """
        cost1 = tf.reduce_sum(tf.pow(mask_pred1 - labels1, 2), 1) + tf.reduce_sum(tf.pow(mask_pred2 - labels2, 2), 1)
        cost1 = tf.reduce_sum(cost1, 1) / tf.squeeze(lengths)
        cost2 = tf.reduce_sum(tf.pow(mask_pred2 - labels1, 2), 1) + tf.reduce_sum(tf.pow(mask_pred1 - labels2, 2), 1)
        cost2 = tf.reduce_sum(cost2, 1) / tf.squeeze(lengths)

        idx = tf.cast(cost1 > cost2, tf.float32) 
        pit_loss = tf.reduce_sum(idx * cost2 + (1 - idx) * cost1)
        """
        front_cost1 = tf.reduce_sum(tf.pow(pred1_front - labels1_front, 2), 1) + tf.reduce_sum(tf.pow(pred2_front - labels2_front, 2), 1)
        front_cost2 = tf.reduce_sum(tf.pow(pred2_front - labels1_front, 2), 1) + tf.reduce_sum(tf.pow(pred1_front - labels2_front, 2), 1)
        first_permu_idx = tf.cast(front_cost1 > front_cost2, tf.float32) 
        cost1_loss = first_permu_idx * front_cost2 + (1 - first_permu_idx) * front_cost1
        cost1 = tf.reduce_sum(tf.pow(pred1_after - labels1_after, 2), 1) + tf.reduce_sum(tf.pow(pred2_after - labels2_after, 2), 1)
        cost1 = tf.concat([cost1_loss, cost1],1)
        cost1 = tf.reduce_sum(cost1, 1) / tf.squeeze(lengths)
        pit_loss = tf.reduce_sum(cost1)
        
        return pit_loss
    return pit_loss

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