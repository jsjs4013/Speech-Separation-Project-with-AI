import tensorflow as tf
import string
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import multiprocessing
import tensorflow as tf
import numpy as np
import librosa
from scipy import signal
import argparse
from numpy.fft import rfft, irfft
from scipy.io.wavfile import write as wav_write
from functools import partial
import logging

from util.global_function import mkdir_p

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 25
INPUT_SIZE = 129
OUTPUT_SIZE = 129

BASE_PATH = '/home/aimaster/lab_storage/Datasets/LibriMix/MixedData/Libri2Mix/'
#tr_path = './mycode/tfrecords/tr_tfrecord/*.tfrecords' # or tr_one_source_tfrecord
#val_path = './mycode/tfrecords/cv_tfrecord/*.tfrecords'
#tt_path = './mycode/tfrecords/tt_tfrecord/*.tfrecords'

"""
START_MASK = tf.fill([1,129],-1)
END_MASK = tf.fill([1,129],-2)
"""
def data_preprocessing(data, check, input_size=129*2, output_size=129*2):
    if check == 'inputs':
        inputs = tf.slice(data, [0, 0], [-1, input_size//2])
        angle = tf.slice(data, [0, input_size//2], [-1, -1])
        
        return inputs, angle
    
    elif check == 'labels':
        label1 = tf.slice(data, [0, 0], [-1, output_size//2])
        label2 = tf.slice(data, [0, output_size//2], [-1, -1])
        
        return label1, label2

# data_type == test then return test data
def read_tfrecord(example, input_size=129*2, output_size=129*2, check='train', case='mixed'):
    tfrecord_format = (
        {
            'inputs': tf.io.FixedLenSequenceFeature(shape=[input_size], dtype=tf.float32),
            'labels': tf.io.FixedLenSequenceFeature(shape=[output_size], dtype=tf.float32),
            'length': tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.float32),
            'name': tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.string),
#             'genders': tf.io.FixedLenSequenceFeature(shape=[2], dtype=tf.float32, allow_missing=True)
        }
    )
    _, example = tf.io.parse_single_sequence_example(example, sequence_features=tfrecord_format)
    
    if case == 'mixed':
        inputs, angle = data_preprocessing(example["inputs"], 'inputs', input_size)
    #     label1, label2 = data_preprocessing(example["labels"], 'labels', input_size)
    
        tiled = tf.tile(tf.expand_dims(example['length'], 1), [1, input_size])

        if check == "test":
            return inputs, angle, example['labels'], example['name'], example['length']
        return inputs, tf.concat([example['labels'], tiled], 0), example['length']
    
    else:
        if check == "test":
            return example['inputs'], example['labels'], example['name'], example['length']
        return example['inputs'], example['labels'], example['length']

def load_dataset(filenames, input_size=129*2, output_size=129*2, check='train', case='mixed'):
    ignore_order = tf.data.Options()
    
    if check == 'train':
        ignore_order.experimental_deterministic = False  # disable order, increase speed
    else:
        ignore_order.experimental_deterministic = True
    
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    
    if case == 'single':
        input_size=input_size//2
        output_size=output_size//2
    dataset = dataset.map(
        partial(read_tfrecord, input_size=input_size, output_size=output_size, check=check, case=case), num_parallel_calls=AUTOTUNE
    )
    
    return dataset

def get_dataset(filenames, input_size=129*2, output_size=129*2, batch_size=BATCH_SIZE, case='mixed'):
    dataset = load_dataset(filenames, input_size, output_size, case=case)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size)
#     dataset = dataset.padded_batch(batch_size, padded_shapes=(None))
    
    return dataset

def get_dataset_for_test(filenames, input_size=129*2, output_size=129*2,batch_size=BATCH_SIZE, case='mixed'):
    dataset = load_dataset(filenames, input_size, output_size, check='test', case=case)
    dataset = dataset.repeat(1)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size)
#     dataset = dataset.padded_batch(batch_size, padded_shapes=(None))
    
    return dataset

def load_data(tr_path, val_path, tt_path, input_size=129, output_size=129, batch_size=25, case = "mixed"):
    tr_path = tr_path + "/*.tfrecords"
    val_path = val_path + "/*.tfrecords"
    tt_path = tt_path + "/*.tfrecords"
    # Load the data
    FILENAMES_TRAINING = tf.io.gfile.glob(tr_path)
    FILENAMES_VALIDATION = tf.io.gfile.glob(val_path)
    FILENAMES_TEST = tf.io.gfile.glob(tt_path)
    print("Train TFRecord Files:", len(FILENAMES_TRAINING))
    print("Validation TFRecord Files:", len(FILENAMES_VALIDATION))
    print("Test TFRecord Files:", len(FILENAMES_TEST))

    train_dataset = get_dataset(FILENAMES_TRAINING, input_size*2, output_size*2, batch_size=batch_size, case=case)
    valid_dataset = get_dataset(FILENAMES_VALIDATION, input_size*2, output_size*2, batch_size=batch_size, case=case)
    test_dataset = get_dataset_for_test(FILENAMES_TEST, input_size*2, output_size*2, batch_size=batch_size, case=case)

    return train_dataset, valid_dataset, test_dataset
    
def main():
    train_dataset, valid, test = load_data('C:/J_and_J_Research/mycode/tfrecords/tr_tfrecord','C:/J_and_J_Research/mycode/tfrecords/cv_tfrecord','C:/J_and_J_Research/mycode/tfrecords/tt_tfrecord')
    for (batch, (inp, tar, length)) in enumerate(train_dataset):
        print(inp.numpy().shape)
        print(tar.numpy().shape)
        print(length)
        print(inp[1].numpy().shape)
        print(inp[1].numpy().shape)    
        print(inp[2].numpy().shape)    
        print(inp[3].numpy().shape)    
        print(tar[0].numpy().shape)
        print(tar[1].numpy().shape)    
        print(tar[2].numpy().shape)    
        print(tar[3].numpy().shape)
        startMask = tf.cast(tf.fill([tar[:,:].shape[0], 1,258],-1),dtype=tf.float32)
        endMask = tf.cast(tf.fill([tar[:,:].shape[0], 1,258],-2),dtype=tf.float32)
        tar = tf.concat([startMask, tar],1)
        tar = tf.concat([tar,endMask],1)
        tar_inp = tar[:, :-1, :]
        tar_real = tar[:, 1:, :]
        print(tar_inp.numpy().shape)
        print(tar_real.numpy().shape)
        tests = tests[tf.newaxis, tf.newaxis, :, :]
        tests = tf.repeat(tests, 4, 0)
        print(tests)

if __name__ == "__main__":
    main()