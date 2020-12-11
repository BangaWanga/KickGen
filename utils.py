import numpy as np
import os
from multi import data_handler
import wavio as io
import glob
from plotting import line_plot
import sounddevice as sd
import librosa
from pathlib import Path

# Note: Currently only mono


def unify_sample(sample: np.ndarray, length=64 * 128):
    if len(sample) > length:
        sample = sample[:length]
        return sample
    else:
        zeros = np.zeros(length)
        zeros[:len(sample)] = sample
        sample = zeros
    return sample


def importer(path: Path, shape: tuple = (64, 128, 1)):
    try:
        frame, sr = librosa.load(path)
    except Exception as e:
        pass  # ToDo: Different error-handling
    frame = np.array(frame)
    frame = unify_sample(frame, length=np.product(shape))
    frame = frame.reshape(shape)
    return frame


def load_folder(path, maxfiles: int = None):

    all_file_paths = [Path(p) for p in
                  glob.glob(path)
                  if p.endswith('.wav')
                  ]
    samples = []
    while len(samples) < maxfiles:
        file_paths = all_file_paths[:maxfiles]
        all_file_paths = all_file_paths[maxfiles:]
        samples = run_multiprocess(file_paths)
    samples = samples[:maxfiles]
    return samples


def run_multiprocess(arg_list: list):
    h = data_handler()
    n_files = 40  # max files without memory error
    slices = [(s * n_files, (s + 1) * n_files - 1) for s in range(int(len(arg_list) / n_files))]
    samples = []
    for s in slices:
        preprocessed = h.multi_threading(arg_list[s[0]: s[1]], importer)
        samples.extend(preprocessed)
    samples = np.array(samples)
    return samples


class Sample:
    def __init__(self, signal: np.ndarray, sr: int, unify=True, classify: bool=False):
        self.signal: np.ndarray = signal
        if classify:
            self.spectrum = self.calculate_spectrum(normalize(signal))
        self.sr = sr
        self.classification = {}
        if unify:
            self.unify_len()

    def __getitem__(self, attr):
        try:
            return self.classification[attr]
        except:
            print(self.classification)

    def classify(self):
        self.classification['boomyness'] = \
            np.round(np.trapz(self.spectrum[0:200], dx=5))
        self.classification['bite'] = \
            np.round(np.trapz(self.spectrum[2000:6000], dx=5))
        self.classification['mushyness'] = \
            np.round(np.trapz(self.spectrum[200:400], dx=5))
        self.classification['noisyness'] = \
            np.round(self.signaltonoise(self.spectrum), 3)

    def calculate_spectrum(self, signal):
        signal = np.fft.fft(signal)
        signal = np.abs(signal)
        np.seterr(divide='ignore')
        signal = np.log(signal)
        signal = np.abs(signal)
        return signal

    def signaltonoise(self, a, axis=0, ddof=0):
        a = np.asanyarray(a)
        m = a.mean(axis)
        sd = a.std(axis=axis, ddof=ddof)
        return np.where(sd == 0, 0, m / sd)

    def play(self):
        sd.play(self.signal, self.sr)

    def unify_len(self, seconds=.25, sampleRate=44100):  # Only works with 1 Channel
        lenght = int(seconds * sampleRate)
        self.signal = self.unify_sample(self.signal, lenght)

    @staticmethod
    def unify_sample(sample: np.ndarray, length=64*128):
        if len(sample) > length:
            sample = sample[:length]
            return sample
        else:
            zeros = np.zeros(length)
            zeros[:len(sample)] = sample
            sample = zeros
        return sample

    def save(self, filename, sampwidth=3):
        io.write(filename, self.signal, self.sr, sampwidth=sampwidth)


def history_plot(history):
    loss = history.__dict__['history']['loss']
    line_plot(loss)


def normalize(array):
    array = np.nan_to_num(array)
    array = np.true_divide(array, np.max(np.abs(array)))
    return array


def tBoard():

    from tensorboard import program
    tb = program.TensorBoard()

    tb.configure(argv=['--logdir serve tmp'])

    url = tb.launch()
    print(url)
