import os
import numpy as np
import librosa
from tensorflow.keras.utils import Sequence
from collections import namedtuple
from util.global_function import mkdir_p
import threading
from scipy.io.wavfile import write as wav_write
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from tensorflow.keras import backend as Kb
import numpy as np
from importlib import reload
import time
from tensorflow.keras.models import Model, Sequential, load_model
import tensorflow_addons as tfa
from models.Schedulers import GumbelAndKLRatioCallback

class RawForVAEGenerator(Sequence):
    def __init__(self, source, wav_dir, files, sourNum='s1', batch_size=10, shuffle=True):
        self.source = source
        self.wav_dir = wav_dir
        self.files = files
        self.sourNum = sourNum
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
        self.sample_rate = 8000
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.source))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __audioread__(self, path, offset=0.0, duration=None, sample_rate=16000):
        signal = librosa.load(path, sr=self.sample_rate, mono=False, offset=offset, duration=duration)

        return signal[0]
    
    def __padding__(self, data):
        n_batch = len(data)
        max_len = max([d.shape[0] for d in data])
        extrapadding = int(np.ceil(max_len / self.sample_rate) * self.sample_rate)
        pad = np.zeros((n_batch, extrapadding))
        
        for i in range(n_batch):
            pad[i, :data[i].shape[0]] = data[i]
        
        return np.expand_dims(pad, -1)
        
    def __data_generation__(self, source_list):
        wav_list = []
        for name in source_list:
            name = name.strip('\n')
            
            s_wav_name = self.wav_dir + self.files + '/' + self.sourNum + '/' + name
            
            # ------- AUDIO READ -------
            s_wav = (self.__audioread__(s_wav_name,  offset=0.0, duration=None, sample_rate=self.sample_rate))
            # --------------------------
            
            # ------- PADDING -------
#             pad_len = max(len(samples1),len(samples2))
#             pad_s1 = np.concatenate([s1_wav, np.zeros([pad_len - len(s1_wav)])])
            
#             extrapadding = ceil(len(pad_s1) / sample_rate) * sample_rate - len(pad_s1)
#             pad_s1 = np.concatenate([pad_s1, np.zeros([extrapadding - len(pad_s1)])])
#             pad_s2 = np.concatenate([s2_wav, np.zeros([extrapadding - len(s2_wav)])])
            # -----------------------
            
            wav_list.append(s_wav)
        
        return wav_list, wav_list, source_list
            
    
    def __len__(self):
        return int(np.floor(len(self.source) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        source_list = [self.source[k] for k in indexes]
        
        if self.files is not 'tt':
            sour, labels, _ = self.__data_generation__(source_list)
            
            # Get Lengths(K value of each batch)
            lengths = np.array([m.shape[0] for m in sour])
            exp = np.expand_dims(lengths, 1)
            exp = np.expand_dims(exp, -1) # [Batch, 1, 1] (length)
            
            # Padding
            sour_pad = self.__padding__(sour) # [Batch, Time_step, Dimension(=1)]
            label_pad = self.__padding__(labels) # [Batch, Time_step, Dimension(=1)]
            
            return sour_pad, np.concatenate([label_pad, exp], axis=1)
        else:
            sour, labels, name = self.__data_generation__(source_list)
            
            # Get Lengths(K value of each batch)
            lengths = np.array([m.shape[0] for m in sour])
            exp = np.expand_dims(lengths, 1)
            exp = np.expand_dims(exp, -1) # [Batch, 1, 1] (length)
            
            # Padding
            sour_pad = self.__padding__(sour) # [Batch, Time_step, Dimension(=1)]
            
            return sour_pad, exp, name


# Data read
Config = namedtuple('Config',  
field_names="d_ff,     d_kv,     d_model,              dropout, feed_forward_proj, num_layers, init_factor," 
            "layer_norm_epsilon, model_type, num_heads, positional_embedding, n_epochs, vocab_size, relative_attention_num_buckets,"
                "model_path, wav_type, size_type, train_type, loss_type, learning_rate_type, emb_or_dense,"
                "input_size, output_size, batch_size, case, ckpt_path, WAV_DIR, LIST_DIR, "
                "test_wav_dir, is_load_model, load_model_dir, latent_size")
args = Config( 2048      , 64      , 512              , 0.1 , "gated-gelu", 4       , 1.,
            1e-06    , "t5"             , 8 , "absolute" , 250     , 129   , 32,
            "CKPT", "wav8k", "min", "train-360", "mse", "inverse_root", "dense",
            40, 40, 32, 'trace', '/home/aimaster/lab_storage/Librimix/models/min/VQ_VAE_4096_gumbel_kl_scheduler_s1_real/', 
            '/home/aimaster/lab_storage/Librimix/LibriMix/MixedData/Libri2Mix/wav8k/min/', 
            '/home/aimaster/lab_storage/Librimix/LibriMix/MixedData/Libri2Mix/wav8k/min/lists/',
            '/home/aimaster/lab_storage/Librimix/models/result/VQ_VAE_valid_best_result/',
            True, '/home/aimaster/lab_storage/Librimix/models/min/VQ_VAE_4096_gumbels_1.0_kl0.2/', 4096)

WAV_DIR = args.WAV_DIR
LIST_DIR = args.LIST_DIR

# Directory List file create
wav_dir = WAV_DIR
output_lst = LIST_DIR
mkdir_p(output_lst)

for folder in ['train-360','dev','test']:
    target = '/s1'
    if folder == 'train-360':
        target = '/s1_100'
    wav_files = os.listdir(wav_dir + folder + target)
    output_lst_files = output_lst + folder + '_wav.lst'
    with open(output_lst_files, 'w') as f:
        for file in wav_files:
            f.write(file + "\n")

print("Generate wav file to .lst done!")

train_dataset = 0
valid_dataset = 0
test_dataset = 0

name_list = []
for files in ['train-360','dev','test']:
    # --- Lead lst file ---
    output_lst_files = LIST_DIR + files + '_wav.lst'
    fid = open(output_lst_files, 'r')
    lines = fid.readlines()
    fid.close()
    # ---------------------
    
    if files == 'train-360':
        train_dataset = RawForVAEGenerator(lines, WAV_DIR, files, 's1_100', args.batch_size)
    elif files == 'dev':
        valid_dataset = RawForVAEGenerator(lines, WAV_DIR, files, 's1', args.batch_size)
    else:
        test_batch = 1
        test_dataset = RawForVAEGenerator(lines, WAV_DIR, files, 's1', test_batch)

ckpt_path = args.ckpt_path
mkdir_p(ckpt_path) # model check point 폴더 만드는 코드

filepath = ckpt_path + "/CKP_ep_{epoch:d}__loss_{val_loss:.5f}_.h5"

initial_learning_rate = 0.001

# learning rate를 점점 줄이는 부분
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True
)

# validation loss에 대해서 좋은 것만 저장됨
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True, mode='min'
)

# early stop 하는 부분인데, validation loss에 대해서 제일 좋은 모델이 저장됨
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True
)

class GumbelSoftmax(layers.Layer):
    def __init__(self, hard=False, name = 'gumbel_softmax',**kwargs):
        super(GumbelSoftmax, self).__init__(name=name, **kwargs)
        
        self.hard = hard
    
    def sample_gumbel(self, shape, eps=1e-8): 
        """Sample from Gumbel(0, 1)"""
        U = tf.random.uniform(shape,minval=0,maxval=1)
        
        return -tf.math.log(-tf.math.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature): 
        """ Draw a sample from the Gumbel-Softmax distribution"""
        y = logits + self.sample_gumbel(tf.shape(logits))
        
        return tf.nn.softmax(y / temperature)
    

    def call(self, inputs, temperature):
        y = self.gumbel_softmax_sample(inputs, temperature)
        
        if self.hard:
            y_hard = tf.cast(tf.equal(y, tf.math.reduce_max(y, 2, keepdims=True)), y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        
        return y


class Encoder(layers.Layer):
    def __init__(self, latent_dim, name = 'encoder',**kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        
        self.conv1d_1 = layers.Conv1D(filters=32, kernel_size=4, strides=2, padding='same', activation='relu')
        self.conv1d_2 = layers.Conv1D(filters=64, kernel_size=4, strides=2, padding='same', activation='relu')
        self.conv1d_3 = layers.Conv1D(filters=128, kernel_size=4, strides=2, padding='same', activation='relu')
        self.conv1d_4 = layers.Conv1D(filters=256, kernel_size=4, strides=2, padding='same', activation='relu')
        self.logit = layers.Conv1D(filters=latent_dim, kernel_size=4, strides=2, activation=None, padding='same')
    
    def call(self, inputs):
        x = self.conv1d_1(inputs)
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        x = self.conv1d_4(x)
        logit = self.logit(x)
        
        return logit


class Decoder(layers.Layer):
    def __init__(self, latent_dim, name = 'decoder',**kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        
        self.trans_conv1d_1 = layers.Conv1DTranspose(filters=256, kernel_size=4, strides=2, activation='relu', padding='same')
        self.trans_conv1d_2 = layers.Conv1DTranspose(filters=128, kernel_size=4, strides=2, activation='relu', padding='same')
        self.trans_conv1d_3 = layers.Conv1DTranspose(filters=64, kernel_size=4, strides=2, activation='relu', padding='same')
        self.trans_conv1d_4 = layers.Conv1DTranspose(filters=32, kernel_size=4, strides=2, activation='relu', padding='same')
        self.logit = layers.Conv1DTranspose(filters=1, kernel_size=4, strides=2, padding='same', activation=None)
    
    def call(self, inputs):
        x = self.trans_conv1d_1(inputs)
        x = self.trans_conv1d_2(x)
        x = self.trans_conv1d_3(x)
        x = self.trans_conv1d_4(x)
        logit = self.logit(x)
        
        return logit

# Custom Metric Si-sdr

class SiSdr(keras.metrics.Metric):
    def __init__(self, name="Si-sdr", **kwargs):
        super(SiSdr, self).__init__(name=name, **kwargs)
        self.sdr = self.add_weight(name="sdr", initializer="zeros")
        self.count = self.add_weight(name="cnt", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        ori_length = tf.shape(y_true)[1]
        
        # Label & Length divide
        labels = tf.slice(y_true, [0, 0, 0], [-1, ori_length-1, -1]) # [batch_size, length_size, 1]
        lengths = tf.slice(y_true, [0, ori_length-1, 0], [-1, -1, 1]) # [batch_size, 1, 1]
        
        # Check sequence length
        batch_size = tf.shape(labels)[0]
        label_size = tf.shape(labels)[1]
        pred_size = tf.shape(y_pred)[1]
        feature_size = tf.shape(labels)[-1]
        
        # Change sequence length
        if label_size < pred_size:
            y_pred = tf.slice(y_pred, [0, 0, 0], [-1, label_size, -1])
        elif label_size > pred_size:
            labels = tf.slice(labels, [0, 0, 0], [-1, pred_size, -1])

        # SI-SDR
        target = tf.linalg.matmul(y_pred, labels, transpose_a=True) * labels / tf.expand_dims(tf.experimental.numpy.square(tf.norm(labels, axis=1)), axis=-1)
        noise = y_pred - target
        values = 10 * tf.experimental.numpy.log10(tf.experimental.numpy.square(tf.norm(target, axis=1)) / tf.experimental.numpy.square(tf.norm(noise, axis=1)))
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            values = tf.multiply(values, sample_weight)
        self.sdr.assign_add(tf.reduce_sum(values))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.sdr / self.count

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.sdr.assign(0.0)
        self.count.assign(0.0)

# Custom loss
# Custom mse
def custom_mse(y_true, y_pred):
    ori_length = tf.shape(y_true)[1]

    # Label & Length divide
    labels = tf.slice(y_true, [0, 0, 0], [-1, ori_length-1, -1]) # [batch_size, length_size, 129]
    lengths = tf.slice(y_true, [0, ori_length-1, 0], [-1, -1, 1]) # [batch_size, 1, 1]
    y_pred_len = tf.shape(y_pred)[1]
    labels_len = tf.shape(labels)[1]
    """if y_pred_len < labels_len :
        labels = labels[:,:y_pred_len,:]
    elif labels_len < y_pred_len :
        y_pred = y_pred[:,:labels_len,:]"""
    loss = tf.reduce_sum(tf.pow(y_pred - labels, 2), axis=[1, 2])
    loss = tf.reduce_mean(loss)

    return loss


# Custom si-sdr loss
def custom_sisdr_loss(y_true, y_pred):
    ori_length = tf.shape(y_true)[1]

    # Label & Length divide
    labels = tf.slice(y_true, [0, 0, 0], [-1, ori_length-1, -1]) # [batch_size, length_size, 1]
    lengths = tf.slice(y_true, [0, ori_length-1, 0], [-1, -1, 1]) # [batch_size, 1, 1]

    target = tf.linalg.matmul(y_pred, labels, transpose_a=True) * labels / tf.expand_dims(tf.experimental.numpy.square(tf.norm(labels, axis=1)), axis=-1)
    noise = y_pred - target
    si_sdr = 10 * tf.experimental.numpy.log10(tf.experimental.numpy.square(tf.norm(target, axis=1)) / tf.experimental.numpy.square(tf.norm(noise, axis=1)))
    si_sdr = tf.reduce_mean(tf.pow(si_sdr, 2))

    return si_sdr

class Vq_vae(keras.Model):
    def __init__(self, latent_dim, temperature=1.0, gumbel_hard=False, name='vqvae', **kwargs):
        super(Vq_vae, self).__init__(name=name, **kwargs)
        self.kl_weight = self.add_weight(name="kl_ratio", initializer="zeros", trainable=False)
        self.temper_weight = self.add_weight(name="temperature", initializer="zeros", trainable=False)
        self.temper_weight.assign_add(temperature + 0.06)
        
        self.latent_dim = latent_dim
        self.softmax = layers.Softmax(-1)
        
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.gumbel = GumbelSoftmax(hard=gumbel_hard)
        
    def call(self, inputs, load=False):
        if load:
            inputs = layers.Input(shape=(None, 1))
        
        
        encode = self.encoder(inputs)
        gumbel = self.gumbel(encode, self.temper_weight)
        decode = self.decoder(gumbel)
        
        # ------------------ KL loss ------------------
        qy = self.softmax(encode)
        log_qy = tf.math.log(qy + 1e-8)
        log_uniform = qy * (log_qy - tf.math.log(1.0 / self.latent_dim))
        kl_loss = tf.reduce_sum(log_uniform, axis=[1, 2])
        kl_loss = tf.reduce_mean(kl_loss) * self.kl_weight
        # ---------------------------------------------
        
        self.add_loss(kl_loss)
        
        return decode
    


latent_size = args.latent_size
epoch = args.n_epochs
BATCH_SIZE = args.batch_size

strategy = tf.distribute.MirroredStrategy()
print('장치의 수: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    model_path = 'CKP_ep_80__loss_20.95814_.h5'
    
    loss_fun = custom_mse
    
    #loss_fun = custom_sisdr_loss
    
    vq_vae = Vq_vae(latent_size, gumbel_hard=False)
    #scheduler = GumbelKLSchedule(1e-5, vq_vae)
    #optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999,epsilon=1e-8, weight_decay = 0.01)
    optimizer = keras.optimizers.Adam(learning_rate=1e-5)
    vq_vae.compile(optimizer, loss=loss_fun, metrics=[SiSdr()])
    
    vq_vae(0, True)
    vq_vae.summary()
    
    # 사용 안할 때는 load_model 주석 처리 하자
    vq_vae.load_weights(args.load_model_dir + model_path)
    # ----------------------------------------
    
    tf.executing_eagerly()

history = vq_vae.fit(
    train_dataset,
    epochs=epoch,
    validation_data=valid_dataset,
    shuffle=True,
    callbacks=[checkpoint_cb,
    GumbelAndKLRatioCallback()],
)

test_wav_dir = args.test_wav_dir
mkdir_p(test_wav_dir) # Result wav 폴더 만드는 코드
def audiowrite(data, path, samplerate=16000, normalize=False, threaded=True):
    """ Write the audio data ``data`` to the wav file ``path``
    The file can be written in a threaded mode. In this case, the writing
    process will be started at a separate thread. Consequently, the file will
    not be written when this function exits.
    :param data: A numpy array with the audio data
    :param path: The wav file the data should be written to
    :param samplerate: Samplerate of the audio data
    :param normalize: Normalize the audio first so that the values are within
        the range of [INTMIN, INTMAX]. E.g. no clipping occurs
    :param threaded: If true, the write process will be started as a separate
        thread
    :return: The number of clipped samples
    """
    data = data.copy()
    int16_max = np.iinfo(np.int16).max
    int16_min = np.iinfo(np.int16).min

    if normalize:
        if not data.dtype.kind == 'f':
            data = data.astype(np.float)
        data /= np.max(np.abs(data))

    if data.dtype.kind == 'f':
        data *= int16_max

    sample_to_clip = np.sum(data > int16_max)
    if sample_to_clip > 0:
        print('Warning, clipping {} samples'.format(sample_to_clip))
    data = np.clip(data, int16_min, int16_max)
    data = data.astype(np.int16)

    if threaded:
        threading.Thread(target=wav_write, args=(path, samplerate, data)).start()
    else:
        wav_write(path, samplerate, data)

    return sample_to_clip

"""with tf.device('/gpu:0'):
    latent_size = args.latent_size
    sample_rate = 8000
    model_path = args.load_model_dir + 'CKP_ep_23__loss_38.72829_.h5'
    
    vq_vae = Vq_vae(latent_size)
    vq_vae(0, True)
    vq_vae.summary()
    vq_vae.load_weights(model_path)

    for batch in test_dataset:
        input_batch, length_batch, name = batch

        result = vq_vae.predict(input_batch)
        
        wav_name = test_wav_dir + name[0][:-5] + '_s1.wav'
        audiowrite(result[0], wav_name, sample_rate, True, True)"""