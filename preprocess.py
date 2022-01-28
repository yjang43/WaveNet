import os
# import soundfile as sf
# import librosa
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import numpy as np

from argparse import ArgumentParser
from tqdm import tqdm


def set_option():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset_path', default='./dataset', type=str)
    parser.add_argument('--audio_type', default='wav', type=str)

    return parser.parse_args()


# audio quantization proposed in the paper

def mu_law_companding_transformation(x, mu=255):
    return np.sign(x) * (np.log(1 + mu * np.abs(x)) / np.log(1 + mu))


def inverse_mu_law_companding_transformation(y, mu=255):
    return np.sign(y) * (((1 + mu) ** np.abs(y) - 1) / mu)

# quantize to 8bit
def quantize(wav, bit):
    wav = mu_law_companding_transformation(wav, 2**bit - 1)
    return ((wav + 1) * 2**(bit - 1)).astype(int)

# recover to 16bit
def inv_quantize(wav, bit):
    wav = (wav / 2**(bit - 1) - 1).astype(float)
    return inverse_mu_law_companding_transformation(wav, bit**2 - 1)



# recursively locate all the wav files

def get_files(dir_, audio_type='wav'):
    def _get_files(fps, dir_):
        _, ds, fs = next(os.walk(dir_))
        for f in fs:
            if f.split('.')[-1] == audio_type:
                fps.append(os.path.join(dir_, f))
        for d in ds:
            _get_files(fps, os.path.join(dir_, d))
            
    files = []
    _get_files(files, dir_)
    return files


# create dataset in ndarray format
# preprocess applied accordingly

def create_dataset(dataset_path, files, sr=11025):
    dataset = []
    for f in tqdm(files):
        # wav, _ = librosa.load(f, sr=sr)
        # wav, _ = sf.read(f, sr=sr)
        orig_sr, wav = wavfile.read(f)
        new_sr = sr
        num_sample = round(len(wav) * new_sr / orig_sr)
        wav = signal.resample(wav, num_sample)
#         wav = librosa.util.normalize(wav)
        quantized_wav = quantize(wav, 8)    # 8 bit
        quantized_wav = np.clip(quantized_wav, 0, 2**8 - 1)
        dataset.append(quantized_wav)
    np.savez(dataset_path, *dataset)


if __name__ == '__main__':
    opt = set_option()

    files = get_files(opt.data_dir, opt.audio_type)
    create_dataset(opt.dataset_path, files)
