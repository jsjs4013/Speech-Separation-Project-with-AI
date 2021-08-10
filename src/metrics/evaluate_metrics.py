import soundfile as sf
import museval
import os
import librosa
import numpy as np
from tqdm.auto import tqdm

def wavread(path):
    wav, sample_rate = sf.read(path, dtype='float32')

#     wav, sample_rate = librosa.load(path, sr=8000, mono=False)
    
    return wav, sample_rate

def pow_np_norm(signal):
    # Compute 2 Norm
    
    return np.square(np.linalg.norm(signal, ord=2))

def pow_norm(s1, s2):
    return np.sum(s1 * s2)

def si_sdr(original, estimated):
    target = pow_norm(estimated, original) * original / pow_np_norm(original)
    noise = estimated - target
    
    return 10 * np.log10(pow_np_norm(target) / pow_np_norm(noise))

def permute_si_sdr(ref1, ref2, est1, est2):
    sdr1 = si_sdr(ref1, est1) + si_sdr(ref2, est2)
    sdr2 = si_sdr(ref1, est2) + si_sdr(ref2, est1)
    if sdr1 > sdr2:
        return sdr1 * 0.5
    else:
        return sdr2 * 0.5

def eval_si_sdr(wav_dir, test_dir):
    wav_files = os.listdir(wav_dir + '/mix_clean')

    si_sdr_list = []
    progress_bar = tqdm(range(len(wav_files)))
    for name in wav_files:
        ref_s1, _ = wavread(wav_dir + '/s1/' + name)
        ref_s2, _ = wavread(wav_dir + '/s2/' + name)
        est_s1, _ = wavread(test_dir + name[:-4] + '_s1.wav')
        est_s2, _ = wavread(test_dir + name[:-4] + '_s2.wav')

        min_len = min(np.size(ref_s1), np.size(est_s1))
        ref_s1, ref_s2 = ref_s1[:min_len], ref_s2[:min_len]
        est_s1, est_s2 = est_s1[:min_len], est_s2[:min_len]

        si_sdr_value = permute_si_sdr(ref_s1, ref_s2, est_s1, est_s2)
        si_sdr_list.append(si_sdr_value)
        progress_bar.update(1)

    mean_si_sdr = np.mean(np.array(si_sdr_list))

    return mean_si_sdr

def eval_sdr(wav_dir, test_dir):
    wav_files = os.listdir(wav_dir + '/mix_clean')

    si_sdr_list = []
    progress_bar = tqdm(range(len(wav_files)))
    for name in wav_files:
        ref_s1, _ = wavread(wav_dir + '/s1/' + name)
        ref_s2, _ = wavread(wav_dir + '/s2/' + name)
        est_s1, _ = wavread(test_dir + name[:-4] + '_s1.wav')
        est_s2, _ = wavread(test_dir + name[:-4] + '_s2.wav')

        min_len = min(np.size(ref_s1), np.size(est_s1))
        ref_s1, ref_s2 = ref_s1[:min_len], ref_s2[:min_len]
        est_s1, est_s2 = est_s1[:min_len], est_s2[:min_len]

        ref_s1 = ref_s1.reshape(-1, 1)
        ref_s2 = ref_s2.reshape(-1, 1)
        est_s1 = est_s1.reshape(-1, 1)
        est_s2 = est_s2.reshape(-1, 1)

        reference_stack = np.stack((ref_s1, ref_s2), axis=0)
        estimated_stack = np.stack((est_s1, est_s2), axis=0)

        (sdr, isr, sir, sar, perm) = museval.metrics.bss_eval(
                reference_stack, estimated_stack, window=np.inf, hop=np.inf,
                compute_permutation=True)

        sdr_back = sdr
        sdr = np.mean(sdr_back)
        if np.isnan(sdr):
            sdr = np.mean(np.nan_to_num(sdr_back))

        si_sdr_list.append(sdr)
        progress_bar.update(1)

    mean_sdr = np.mean(np.array(si_sdr_list))

    return mean_sdr

def main():
    wav_dir = '/home/aimaster/lab_storage/Datasets/LibriMix/MixedData/Libri2Mix/wav8k/min/test/'
    test_dir = '/home/aimaster/lab_storage/models/Librimix/wav8k/min/BASELINE_upit_dropout0.3_epoch132/'
    si_sdr = eval_si_sdr(wav_dir, test_dir)
    sdr = eval_sdr(wav_dir, test_dir)

    print("The SI-SDR (db) :", si_sdr)
    print("The SDR (db) :", sdr)

if __name__ == "__main__" :
    main()