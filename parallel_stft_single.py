import tensorflow as tf
import string

import multiprocessing
import tensorflow as tf
import numpy as np
import librosa
from scipy import signal
import argparse
import os, sys
from numpy.fft import rfft, irfft
from scipy.io.wavfile import write as wav_write

import librosa.display
sys.path.append('.')

import threading


#/home/aimaster/lab_storage/Datasets/LibriMix/MixedData/Libri2Mix/wav8k/max/
#'/home/aimaster/lab_storage/Datasets/LibriMix/MixedData/Libri2Mix/'
#'/home/aimaster/lab_storage/Datasets/LibriMix/MixedData/Libri2Mix/wav8k/max/dev/'
wav_dir = '/home/aimaster/lab_storage/Datasets/LibriMix/MixedData/Libri2Mix/'
list_dir = '/home/aimaster/lab_storage/Datasets/LibriMix/MixedData/Libri2Mix/'
tfrecord_dir = '/home/aimaster/lab_storage/Datasets/LibriMix/MixedData/Libri2Mix/'
gender_list = ''#'./wsj0-train-spkrinfo.txt'
process_num = 8

def mkdir_p(path):
    """ Creates a path recursively without throwing an error if it already exists
    :param path: path to create
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)

def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    """Generate a new array that chops the given array along the given axis into overlapping frames.
    example:
    >>> segment_axis(np.arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])
    arguments:
    a       The array to segment
    length  The length of each frame
    overlap The number of array elements by which the frames should overlap
    axis    The axis to operate on; if None, act on the flattened array
    end     What to do with the last frame, if the array is not evenly
            divisible into pieces. Options are:
            'cut'   Simply discard the extra values
            'wrap'  Copy values from the beginning of the array
            'pad'   Pad with a constant value
    endvalue    The value to use for end='pad'
    The array is not copied unless necessary (either because it is
    unevenly strided and being flattened or because end is set to
    'pad' or 'wrap').
    """

    if axis is None:
        a = np.ravel(a)  # may copy
        axis = 0

    l = a.shape[axis]

    if overlap >= length: raise ValueError(
        "frames cannot overlap by more than 100%")
    if overlap < 0 or length <= 0: raise ValueError(
        "overlap must be nonnegative and length must be positive")

    if l < length or (l - length) % (length - overlap):
        if l > length:
            roundup = length + (1 + (l - length) // (length - overlap)) * (
                    length - overlap)
            rounddown = length + ((l - length) // (length - overlap)) * (
                    length - overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown < l < roundup
        assert roundup == rounddown + (length - overlap) or (
                roundup == length and rounddown == 0)
        a = a.swapaxes(-1, axis)

        if end == 'cut':
            a = a[..., :rounddown]
        elif end in ['pad', 'wrap']:  # copying will be necessary
            s = list(a.shape)
            s[-1] = roundup
            b = np.empty(s, dtype=a.dtype)
            b[..., :l] = a
            if end == 'pad':
                b[..., l:] = endvalue
            elif end == 'wrap':
                b[..., l:] = a[..., :roundup - l]
            a = b

        a = a.swapaxes(-1, axis)

    l = a.shape[axis]
    if l == 0: raise ValueError(
        "Not enough data points to segment array in 'cut' mode; try 'pad' or 'wrap'")
    assert l >= length
    assert (l - length) % (length - overlap) == 0
    n = 1 + (l - length) // (length - overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n, length) + a.shape[axis + 1:]
    newstrides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[axis + 1:]

    if not a.flags.contiguous:
        a = a.copy()
        newstrides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[axis + 1:]
        return np.ndarray.__new__(np.ndarray, strides=newstrides, shape=newshape, buffer=a, dtype=a.dtype)

    try:
        return np.ndarray.__new__(np.ndarray, strides=newstrides, shape=newshape, buffer=a, dtype=a.dtype)
    except TypeError or ValueError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[axis + 1:]
        return np.ndarray.__new__(np.ndarray, strides=newstrides, shape=newshape, buffer=a, dtype=a.dtype)

def _samples_to_stft_frames(samples, size, shift):
    """
    Calculates STFT frames from samples in time domain.
    :param samples: Number of samples in time domain.
    :param size: FFT size.
    :param shift: Hop in samples.
    :return: Number of STFT frames.
    """

    return np.ceil((float(samples) - size + shift) / shift).astype(np.int)

def _stft_frames_to_samples(frames, size, shift):
    """
    Calculates samples in time domain from STFT frames
    :param frames: Number of STFT frames.
    :param size: FFT size.
    :param shift: Hop in samples.
    :return: Number of samples in time domain.
    """
    return frames * shift + size - shift

def stft(time_signal, time_dim=None, size=1024, shift=256,
         window=signal.blackman, fading=True, window_length=None):
    """
    Calculates the short time Fourier transformation of a multi channel multi
    speaker time signal. It is able to add additional zeros for fade-in and
    fade out and should yield an STFT signal which allows perfect
    reconstruction.
    :param time_signal: multi channel time signal.
    :param time_dim: Scalar dim of time.
        Default: None means the biggest dimension
    :param size: Scalar FFT-size.
    :param shift: Scalar FFT-shift. Typically shift is a fraction of size.
    :param window: Window function handle.
    :param fading: Pads the signal with zeros for better reconstruction.
    :param window_length: Sometimes one desires to use a shorter window than
        the fft size. In that case, the window is padded with zeros.
        The default is to use the fft-size as a window size.
    :return: Single channel complex STFT signal
        with dimensions frames times size/2+1.
    """
    if time_dim is None:
        time_dim = np.argmax(time_signal.shape)

    # Pad with zeros to have enough samples for the window function to fade.
    if fading:
        pad = [(0, 0)] * time_signal.ndim
        pad[time_dim] = [size - shift, size - shift]
        time_signal = np.pad(time_signal, pad, mode='constant')

    # Pad with trailing zeros, to have an integral number of frames.
    frames = _samples_to_stft_frames(time_signal.shape[time_dim], size, shift)
    samples = _stft_frames_to_samples(frames, size, shift)
    pad = [(0, 0)] * time_signal.ndim
    pad[time_dim] = [0, samples - time_signal.shape[time_dim]]
    time_signal = np.pad(time_signal, pad, mode='constant')
    

    if window_length is None:
        window = window(size)
    else:
        window = window(window_length)
        window = np.pad(window, (0, size - window_length), mode='constant')

    time_signal_seg = segment_axis(time_signal, size,
                                   size - shift, axis=time_dim)

    letters = string.ascii_lowercase
    mapping = letters[:time_signal_seg.ndim] + ',' + letters[time_dim + 1] \
              + '->' + letters[:time_signal_seg.ndim]

    return rfft(np.einsum(mapping, time_signal_seg, window), axis=time_dim + 1)

def audioread(path, offset=0.0, duration=None, sample_rate=16000):
    """
    Reads a wav file, converts it to 32 bit float values and reshapes accoring
    to the number of channels.
    Now, this is a wrapper of librosa with our common defaults.
    :param path: Absolute or relative file path to audio file.
    :type: String.
    :param offset: Begin of loaded audio.
    :type: Scalar in seconds.
    :param duration: Duration of loaded audio.
    :type: Scalar in seconds.
    :param sample_rate: Sample rate of audio
    :type: scalar in number of samples per second
    :return:
    """
    signal = librosa.load(path, sr=sample_rate, mono=False, offset=offset, duration=duration)
    
    return signal[0]

def max_length(target_wav_dir, name, mix_or_not):
    max_len = 0
    
    for name in lines:
        name = name.strip('\n')

        wav_name = target_wav_dir + '/' + mix_or_not + '/' + name
        audio_wav = audioread(wav_name, offset=0.0, duration=None, sample_rate=sample_rate)
        mix_len = len(audio_wav)

        if mix_len > max_len:
            max_len = mix_len
    
    # 초 맞춰주는 부분
    max_len = ceil(max_len / sample_rate) * sample_rate
    
    return max_len

def make_sequence_example(inputs, labels, name, genders=False):
    input_features = [tf.train.Feature(float_list=tf.train.FloatList(value=input_)) for input_ in inputs]
    label_features = [tf.train.Feature(float_list=tf.train.FloatList(value=label)) for label in labels]
    len_feature = [tf.train.Feature(float_list=tf.train.FloatList(value=[length]))]
    name_feature = [tf.train.Feature(bytes_list=tf.train.BytesList(value=[name.encode('utf-8')]))]
#     gender_features = [tf.train.Feature(float_list=tf.train.FloatList(value=genders))]
    
    feature_list = {
        'inputs': tf.train.FeatureList(feature=input_features),
        'labels': tf.train.FeatureList(feature=label_features),
        'length': tf.train.FeatureList(feature=len_feature),
        'name' : tf.train.FeatureList(feature=name_feature)
#         'genders': tf.train.FeatureList(feature=gender_features)
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    
    return tf.train.SequenceExample(feature_lists=feature_lists)


def gen_feats(wav_name, sample_rate, window_size, window_shift, file, tfrecord_d, target_wav_dir, max_len, case='mixed'):
    mix_wav_name = target_wav_dir + '/mix/' + wav_name
    s1_wav_name  = target_wav_dir + '/s1/' + wav_name
    s2_wav_name  = target_wav_dir + '/s2/' + wav_name

    # value initiallization
    mix_wav = 0
    s1_wav = 0
    s2_wav = 0
    mix_stft = 0
    s1_stft = 0
    s2_stft = 0
    
    if case == 'mixed':
        # ------- AUDIO READ -------
        mix_wav = audioread(mix_wav_name, offset=0.0, duration=None, sample_rate=sample_rate)
        s1_wav  = audioread(s1_wav_name,  offset=0.0, duration=None, sample_rate=sample_rate)
        s2_wav  = audioread(s2_wav_name,  offset=0.0, duration=None, sample_rate=sample_rate)
        # --------------------------
        
        # ------- AUDIO PAD -------
        mix_wav_pad = np.pad(mix_wav, (0, max_len - len(mix_wav)), 'constant', constant_values=(0))
        s1_wav_pad = np.pad(s1_wav, (0, max_len - len(s1_wav)), 'constant', constant_values=(0))
        s2_wav_pad = np.pad(s2_wav, (0, max_len - len(s2_wav)), 'constant', constant_values=(0))
        # -------------------------

        # ------- STFT -------
        mix_stft = stft(mix_wav, time_dim=0, size=window_size, shift=window_shift)
        
        mix_stft_pad = stft(mix_wav_pad, time_dim=0, size=window_size, shift=window_shift)
        s1_stft_pad = stft(s1_wav_pad, time_dim=0, size=window_size, shift=window_shift)
        s2_stft_pad = stft(s2_wav_pad, time_dim=0, size=window_size, shift=window_shift)
        # --------------------
        
        part_name = os.path.splitext(wav_name)[0]
        tfrecords_name = tfrecord_d + file + '_tfrecord/' + part_name + '.tfrecords'
        
        with tf.io.TFRecordWriter(tfrecords_name) as writer:
            tf.compat.v1.logging.info("Writing utterance %s" %tfrecords_name)

            mix_abs = np.abs(mix_stft_pad)
            mix_angle = np.angle(mix_stft_pad)

            s1_abs = np.abs(s1_stft_pad)
            s1_angle = np.angle(s1_stft_pad)

            s2_abs = np.abs(s2_stft_pad)
            s2_angle = np.angle(s2_stft_pad)

            inputs = np.concatenate((mix_abs, mix_angle), axis=1)
            labels = np.concatenate((s1_abs * np.cos(mix_angle - s1_angle), s2_abs * np.cos(mix_angle - s2_angle)), axis=1)
            
            ex = make_sequence_example(inputs, labels, mix_stft.shape[0], part_name)
            writer.write(ex.SerializeToString())
    else:
        # ------- AUDIO READ -------
        s1_wav  = audioread(s1_wav_name,  offset=0.0, duration=None, sample_rate=sample_rate)
        s2_wav  = audioread(s2_wav_name,  offset=0.0, duration=None, sample_rate=sample_rate)
        # --------------------------
        
        # ------- AUDIO PAD -------
        s1_wav_pad = np.pad(s1_wav, (0, max_len - len(s1_wav)), 'constant', constant_values=(0))
        s2_wav_pad = np.pad(s2_wav, (0, max_len - len(s2_wav)), 'constant', constant_values=(0))
        # -------------------------
        
        # ------- STFT -------
        s1_stft  = stft(s1_wav,  time_dim=0, size=window_size, shift=window_shift)
        s2_stft  = stft(s2_wav,  time_dim=0, size=window_size, shift=window_shift)
        
        s1_stft_pad = stft(s1_wav_pad, time_dim=0, size=window_size, shift=window_shift)
        s2_stft_pad = stft(s2_wav_pad, time_dim=0, size=window_size, shift=window_shift)
        # --------------------
        
        part_name = os.path.splitext(wav_name)[0]
        tfrecords_s1_name = tfrecord_d + file + '_one_source_tfrecord/' + part_name + '_s1.tfrecords'
        tfrecords_s2_name = tfrecord_d + file + '_one_source_tfrecord/' + part_name + '_s2.tfrecords'
        
        with tf.io.TFRecordWriter(tfrecords_s1_name) as writer:
            tf.compat.v1.logging.info("Writing utterance %s" %tfrecords_s1_name)
            
            s1_abs = np.abs(s1_stft_pad)
            s1_angle = np.angle(s1_stft_pad)
            
            ex = make_sequence_example(s1_abs, s1_angle, s1_stft.shape[0], part_name + '_s1')
            writer.write(ex.SerializeToString())

        with tf.io.TFRecordWriter(tfrecords_s2_name) as writer:
            tf.compat.v1.logging.info("Writing utterance %s" %tfrecords_s2_name)
            
            s2_abs = np.abs(s2_stft_pad)
            s2_angle = np.angle(s2_stft_pad)
            
            ex = make_sequence_example(s2_abs, s2_angle, s2_stft.shape[0], part_name + '_s2')
            writer.write(ex.SerializeToString())

def gen_feats_total(lines, sample_rate, window_size, window_shift, files, tfrecord_d, target_wav_dir, max_len, CASE):
    for name in lines:
        name = name.strip('\n')
        
        gen_feats(name, sample_rate, window_size, window_shift, files, tfrecord_d, target_wav_dir, max_len, CASE)

        
def main():
    CASE = 'signal' # mixed or signal
    
    sample_rate = 8000
    window_size = 256
    window_shift = 128
    threads = []

    for wave_select in ['wav16k/', 'wav8k/']:
        for big_folder in ['max/','min/']:
            for files in ['dev', 'test', 'train-100','train-360']:
                tfrecord_dir = wav_dir + wave_select + big_folder + files + "/"
                output_lst_files = list_dir + wave_select + big_folder + files + '_wav.lst'
                target_wav_dir = wav_dir + wave_select + big_folder + files + "/"
                
                # Read file list
                fid = open(output_lst_files, 'r')
                lines = fid.readlines()
                fid.close()
                
                
                # Check max length
                max_len = 0
                
                if CASE == 'mixed':
                    for name in lines:
                        name = name.strip('\n')
                        max_len = max_length(target_wav_dir, name, 'mix')
                else:
                    for name in lines:
                        name = name.strip('\n')
                        max_len1 = max_length(target_wav_dir, name, 's1')
                        max_len2 = max_length(target_wav_dir, name, 's2')

                    if max_len1 >= max_len2:
                        max_len = max_len1
                    else:
                        max_len = max_len2
                
                
                # Make tfrecords files
                mkdir_p(tfrecord_dir + files + '_tfrecord') # tfrecord_dir 폴더 만드는 코드
                mkdir_p(tfrecord_dir + files + '_one_source_tfrecord') # tfrecord_dir 폴더 만드는 코드
                threads.append(threading.Thread(target=gen_feats_total, args=(lines, sample_rate, window_size, window_shift, files, tfrecord_dir, target_wav_dir, max_len, CASE)))
    

    for thread in threads :
        thread.start()
    print('Done')

if __name__ == "__main__":
    main()