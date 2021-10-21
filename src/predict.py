from tqdm.auto import tqdm
from tensorflow.keras.utils import Sequence
import numpy as np
import tensorflow as tf
from collections import namedtuple
from util.global_function import mkdir_p
import os
from util.math_function import create_padding_mask, create_look_ahead_mask
from losses.custom_loss import mse_with_proper_loss, MSE_Custom_Loss_No_Length, pit_with_outputsize, pit_with_stft_trace
import librosa
from models.Schedulers import CustomSchedule
from models.Real_Layers import T5Model, T5ModelNoMaskCreationModel, T5ModelYesMaskCreationModel
from pre_processing.data_pre_processing import load_data
from models.T5_variations import T5ChangedSTFT

def create_masks(inp, tar, length=None):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp, length) # (batch, 1, 1, seq_len)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp, length)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1]) # (seq_len, seq_len)
    dec_target_padding_mask = create_padding_mask(tar, length) # (batch, 1, 1, seq_len)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask) # (batch, 1, seq_len, seq_len)

    return enc_padding_mask, combined_mask, dec_padding_mask

class RawDataGenerator(Sequence):
    def __init__(self, Mix, wav_dir, files, batch_size=10, shuffle=True):
        self.Mix = Mix
        self.wav_dir = wav_dir
        self.files = files
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.Mix))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __audioread__(self, path, offset=0.0, duration=None, sample_rate=16000):
        signal = librosa.load(path, sr=sample_rate, mono=False, offset=offset, duration=duration)

        return signal[0]
    
    def __padding__(self, data):
        n_batch = len(data)
        max_len = max([d.shape[0] for d in data])
        pad = np.zeros((n_batch, max_len, data[0].shape[1]))
        
        for i in range(n_batch):
            pad[i, :data[i].shape[0]] = data[i]
        
        return pad
        
    def __data_generation__(self, Mix_list):
        sample_rate = 8000
        L = 40
        
        mix_wav_list = []
        label_wav_list = []
        for name in Mix_list:
            name = name.strip('\n')
            
            s1_wav_name = self.wav_dir + self.files + '/s1/' + name
            
            # ------- AUDIO READ -------
            s1_wav = (self.__audioread__(s1_wav_name,  offset=0.0, duration=None, sample_rate=sample_rate))
            # --------------------------
            
            # ------- TIME AXIS CALCULATE -------
            K = int(np.ceil(len(s1_wav) / L))
            # -----------------------------------
            
            # ------- PADDING -------
            pad_len = K * L
            pad_s1 = np.concatenate([s1_wav, np.zeros([pad_len - len(s1_wav)])])
            # -----------------------
            
            # ------- RESHAPE -------
            s1 = np.reshape(pad_s1, [K, L])
            # -----------------------
            
            # ------- CONCAT S1 S2 -------
            # ----------------------------
            
            label_wav_list.append(s1)
        
        return label_wav_list, label_wav_list

    
    def __len__(self):
        return int(np.floor(len(self.Mix) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        Mix_list = [self.Mix[k] for k in indexes]
        
        if self.files is not 'test':
            mix, labels = self.__data_generation__(Mix_list)
            
            # Get Lengths(K value of each batch)
            lengths = np.array([m.shape[0] for m in mix])
            tiled = np.tile(np.expand_dims(lengths, 1), [1, labels[0].shape[1]])
            tiled = np.expand_dims(tiled, 1)
            
            # Padding
            mix_pad = self.__padding__(mix) # [Batch, Time_step, Dimension]
            label_pad = self.__padding__(labels) # [Batch, Time_step, Dimension * 2]
            
            return mix_pad, np.concatenate([label_pad, tiled], axis=1), lengths
        else:
            mix, labels = self.__data_generation__(Mix_list)
            
            # Get Lengths(K value of each batch)
            lengths = np.array([m.shape[0] for m in mix])
            tiled = np.tile(np.expand_dims(lengths, 1), [1, labels[0].shape[1]])
            tiled = np.expand_dims(tiled, 1)
            
            # Padding
            mix_pad = self.__padding__(mix) # [Batch, Time_step, Dimension]
            
            return mix_pad, tiled, lengths, Mix_list


def build_real_T5(input_size, output_size, args):
    inputs = (tf.keras.layers.Input(shape=(None, input_size)),
    tf.keras.layers.Input(shape=(None, output_size)),
    tf.keras.layers.Input(shape=(1)) )
    # targets, length
        
    transformer = T5ModelNoMaskCreationModel(num_layers=args.num_layers, d_model=args.d_model, num_heads=args.num_heads, d_ff=args.d_ff, d_kv = args.d_kv, vocab_size=0, feed_forward_proj = args.feed_forward_proj, 
            relative_attention_num_buckets=args.relative_attention_num_buckets, eps=args.layer_norm_epsilon, dropout=args.dropout, factor=args.init_factor,
            embed_or_dense="conv", target_size=output_size)

    inp, tar, length = inputs
    #dec_padding_mask = tf.squeeze(dec_padding_mask)
    #outputs = tf.keras.layers.Conv1D(filters=129, kernel_size=2, activation = 'sigmoid', padding='same')(inp)

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar, length)
    enc_padding_mask = tf.squeeze(enc_padding_mask)
    outputs = transformer(input_ids=inp, attention_mask=enc_padding_mask, 
            decoder_input_ids=tar, 
             training=False) # (batch_size, tar_seq_len, target_vocab_size)
    
    model = T5ChangedSTFT(inputs=inputs, outputs=outputs)
    model.summary()
    learning_rate = CustomSchedule(args.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,epsilon=1e-8)
    #optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999,epsilon=1e-8, weight_decay = 0.01)
    #model.add_metric(tf.keras.metrics.Mean(name='train_loss')(outputs))
    #model.compile(loss=mse_with_proper_loss(output_size), optimizer=optimizer)
    model.compile(loss=pit_with_stft_trace(output_size), optimizer=optimizer)
#     model.compile(loss=keras.losses.mean_squared_error, optimizer=adam)

    return model


def evaluate(inp, transformer, output, length=None, max_length=1800):
    # inp sentence is portuguese, hence adding the start and end token
    encoder_input = inp
    max_length = inp.shape[1]
    cur_length = output.shape[1]
    batch_size = inp.shape[0]
    
    # as the target is english, the first word to the transformer should be the
    # english start token.
    #startMask = tf.cast(tf.fill([1,258],-1),dtype=tf.float32)
    #output = tf.expand_dims(startMask, 0)
    #output = tf.repeat(output, batch_size, 0)
    #zero_clipping = tf.constant([0.])
    output = tf.cast(output,tf.float32)
    print("output:",output)

    progress_bar = tqdm(range(max_length-cur_length))
    for i in range(max_length-cur_length):
        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions = transformer((encoder_input, output, length), training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
        #predictions = tf.math.maximum(predictions, zero_clipping)
        predicted_id = predictions

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=1)
        progress_bar.update(1)

    return output

Config = namedtuple('Config',  
field_names="d_ff,     d_kv,     d_model,              dropout, feed_forward_proj, num_layers, init_factor," 
            "layer_norm_epsilon, model_type, num_heads, positional_embedding, n_epochs, vocab_size, relative_attention_num_buckets,"
                "model_path, wav_type, size_type, train_type, loss_type, learning_rate_type, emb_or_dense,"
                "input_size, output_size, batch_size, case, ckpt_path, WAV_DIR, LIST_DIR, "
                "test_wav_dir, is_load_model")
args = Config( 2048      , 64      , 512              , 0.1 , "gated-gelu", 4       , 1.,
            1e-06    , "t5"             , 8 , "absolute" , 200     , 129   , 32,
            "CKPT", "wav8k", "min", "train-360", "mse", "inverse_root", "dense",
            40, 40, 8, 'trace', '/home/aimaster/lab_storage/Librimix/models/min/T5_autoencoder_conv1d', 
            '/home/aimaster/lab_storage/Librimix/LibriMix/MixedData/Libri2Mix/wav8k/min/', 
            '/home/aimaster/lab_storage/Librimix/LibriMix/MixedData/Libri2Mix/wav8k/min/lists/',
            '/home/aimaster/lab_storage/Librimix/models/result/t5_autoencoder3',
            True)

WAV_DIR = args.WAV_DIR
LIST_DIR = args.LIST_DIR

# Directory List file create
wav_dir = WAV_DIR
output_lst = LIST_DIR
mkdir_p(output_lst)

for folder in ['test']:
    wav_files = os.listdir(wav_dir + folder + "/s1")
    output_lst_files = output_lst + folder + '_wav.lst'
    with open(output_lst_files, 'w') as f:
        for file in wav_files:
            f.write(file + "\n")

print("Generate wav file to .lst done!")

name_list = []
for files in ['test']:
    # --- Lead lst file ---
    output_lst_files = LIST_DIR + files + '_wav.lst'
    fid = open(output_lst_files, 'r')
    lines = fid.readlines()
    fid.close()
    # ---------------------
    
    if files == 'train-360':
        train_dataset = RawDataGenerator(lines, WAV_DIR, files, args.batch_size)
    elif files == 'dev':
        valid_dataset = RawDataGenerator(lines, WAV_DIR, files, args.batch_size)
    else:
        test_dataset = RawDataGenerator(lines, WAV_DIR, files, args.batch_size)


from scipy.io.wavfile import write as wav_write
import threading, sys, os
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
            
mkdir_p(args.test_wav_dir)
with tf.device('/gpu:0'):
    ckpt_path = args.ckpt_path
    model_path = ckpt_path + "/CKP_ep_13__loss_0.00005_.h5"
    sample_rate = 8000

    # T5 load
    if True:
        model = build_real_T5(args.input_size, args.output_size, args)
        model.load_weights(model_path)
    # baseline
    if False:
        model = load_model(model_path, custom_objects={'pit_loss': pit_with_outputsize(OUTPUT_SIZE)})
    cnt = 0
    check = 0
    for batch in test_dataset:
        mix_pad, tiled, lengths, names = batch
        # input_batch, angle_batch, label_batch, name, length
        tf.executing_eagerly() # requires r1.7
        
        
        #startMask = tf.cast(tf.fill([tf.shape(label_batch)[0], 1, tf.shape(label_batch)[-1]],-1),dtype=tf.float32)
        #tar = tf.concat([startMask, label_batch],1)
        max_length = 1800
        output = tf.slice(mix_pad,[0,0,0],[-1,100,-1])
        result = evaluate(mix_pad, model, output, lengths, max_length)
        results = tf.reshape(result,[mix_pad.shape[0],-1,1])

        for i in range(mix_pad.shape[0]):
            wav_name = names[i].strip()
            wav_name1 = args.test_wav_dir + '/' + wav_name + '_s1.wav'
            wav = tf.squeeze(results[i,:,:]).numpy()
            wav = wav[:lengths[i]*40]
            audiowrite(wav, wav_name1, sample_rate, True, True)
        if check == -1:
            break

        if (cnt + 1) % 10 == 0:
            print((cnt + 1) * args.batch_size)

        cnt += 1
