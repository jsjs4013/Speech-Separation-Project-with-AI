import threading, sys, os
import time
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import optimizers
from collections import namedtuple
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import scipy
import numpy as np
from util.global_function import mkdir_p
from util.math_function import create_padding_mask, create_look_ahead_mask
from losses.custom_loss import MSE_Custom_Loss
from Layers import TransformerSpeechSep
from Schedulers import CustomSchedule
from pre_processing.data_pre_processing import load_data

BATCH_SIZE = 25
INPUT_SIZE = 129
OUTPUT_SIZE = 129

"""
combined_mask = create_look_ahead_mask(tf.shape(tar_inp)[1])
combined_mask = combined_mask[tf.newaxis, tf.newaxis, :, :]
combined_mask = tf.repeat(combined_mask, tf.shape(tar_inp)[0], 0)
"""

def create_masks(inp, tar, length):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp, length) # (batch, 1, 1, seq_len)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(tar, length)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1]) # (seq_len, seq_len)
    dec_target_padding_mask = create_padding_mask(tar, length) # (batch, 1, 1, seq_len)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask) # (batch, 1, seq_len, seq_len)

    return enc_padding_mask, combined_mask, dec_padding_mask


def train(args):
    train_dataset, valid_dataset, test_dataset = load_data(args.tr_path, args.val_path, args.tt_path, args.input_size, args.input_size, args.batch_size, args.case)

    ckpt_path = args.ckpt_path
    mkdir_p(ckpt_path) # model check point 폴더 만드는 코드
    filepath = ckpt_path + "/CKP_ep_{epoch:d}__loss_{val_loss:.5f}_.h5"


    loss_object = tf.keras.losses.MeanSquaredError(
        reduction=tf.keras.losses.Reduction.AUTO, name='mean_squared_error'
    )

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    #checkpoint_path = "./checkpoints/train"
    checkpoint_path = ckpt_path

    learning_rate = CustomSchedule(args.d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                        epsilon=1e-9)
    transformer = TransformerSpeechSep(num_layers=args.num_layers, d_model=args.d_model, num_heads=args.num_heads, d_ff=args.d_ff,
            input_size=129, target_size=258,
            pe_input=10000, pe_target=6000, dropout = args.dropout)

    ckpt = tf.train.Checkpoint(transformer=transformer,
                            optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    START_MASK = tf.fill([1,129],-1)
    END_MASK = tf.fill([1,129],-2)
    startMask = tf.cast(START_MASK,dtype=tf.float32)
    endMask = tf.cast(END_MASK,dtype=tf.float32)

    train_step_signature = [
        tf.TensorSpec(shape=(None, None, 129), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, 258), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    ]


    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar, length): # (batch, seq_len, values)
        """print('inp',inp.shape) 
        startMask = tf.cast(tf.fill([1,258],-1),dtype=tf.float32)
        endMask = tf.cast(tf.fill([1,258],-2),dtype=tf.float32)
        tar_inp = tf.concat([startMask, tar],0)
        tar_real = tf.concat([tar, endMask],0)
        print('tar_inp',tar_inp.shape)
        print('tar_real',tar_real[0])"""
        tar_inp = tar[:, :-1, :]
        tar_real = tar[:, 1:, :]
        print('tar_inp',tar_inp.shape)

        """
        enc_padding_mask, combined_mask, dec_padding_mask = None, None, None
        combined_mask = create_look_ahead_mask(tf.shape(tar_inp)[1])
        combined_mask = combined_mask[tf.newaxis, tf.newaxis, :, :]
        combined_mask = tf.repeat(combined_mask, tf.shape(tar_inp)[0], 0)
        """
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp, length)
        
        print(enc_padding_mask, combined_mask, dec_padding_mask)
        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp,
                                        True,
                                        enc_padding_mask,
                                        combined_mask,
                                        dec_padding_mask)
            
            loss = MSE_Custom_Loss(tar_real, predictions, length, loss_object)
            
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        #train_accuracy(accuracy_function(tar_real, predictions))


    # training
    for epoch in range(args.n_epochs):
        start = time.time()

        train_loss.reset_states()
        #train_accuracy.reset_states()

        for (batch, (inp, tar, length)) in enumerate(train_dataset):
            startMask = tf.cast(tf.fill([tar[:,:].shape[0], 1,258],-1),dtype=tf.float32)
            #endMask = tf.cast(tf.fill([tar[:,:].shape[0], 1,258],-2),dtype=tf.float32)
            tar = tf.concat([startMask, tar],1)
            #tar = tf.concat([tar,endMask],1)
            train_step(inp, tar, length)

            if batch % 50 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f}')

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f}')

        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')


def main():
    Config = namedtuple('Config',  
    field_names="d_ff,     d_kv,     d_model,              dropout, feed_forward_activation, num_layers," 
                "layer_norm_epsilon, model_type, num_heads, positional_embedding, n_epochs, vocab_size,"
                    "model_path, wav_type, size_type, train_type, loss_type, learning_rate_type,"
                    "input_size, output_size, batch_size, case, ckpt_path, tr_path, val_path, tt_path")
    args = Config( 2048      , 64      , 512              , 0.1 , "gated_gelu", 4       , 
                1e-06    , "t5"             , 8 , "absolute" , 10     , 129   ,
                "CKPT", "wav8k", "min", "train-360", "mse", "inverse_root",
                129, 258, 25, 'mixed', 'C:/J_and_J_Research/mycode/CKPT', 'C:/J_and_J_Research/mycode/tfrecords/tr_tfrecord', 'C:/J_and_J_Research/mycode/tfrecords/cv_tfrecord', 'C:/J_and_J_Research/mycode/tfrecords/tt_tfrecord')
    print("hello World!")
    train(args)

if __name__ == "__main__":
    main()