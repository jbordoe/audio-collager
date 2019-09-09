from dataclasses import dataclass

import numpy as np
from numpy.linalg import norm
from dtw import accelerated_dtw as dtw

import librosa
import soundfile as sf

class Util:
    @dataclass
    class AudioFile:
        timeseries: np.ndarray
        sample_rate: int
        mfcc: np.ndarray = None

    @staticmethod
    def read_audio(path):
        timeseries, sample_rate = librosa.load(path)
        return Util.AudioFile(timeseries, sample_rate)

    @staticmethod
    def save_audio(audiofile, path):
        sf.write(path, audiofile.timeseries, audiofile.sample_rate, format='wav')

    @staticmethod
    def chop_audio(audiofile, window_size_ms):
        pointer = 0
        timeseries = audiofile.timeseries
        sample_rate = audiofile.sample_rate

        window_size_frames = int((window_size_ms / 1000) * sample_rate)
        slices = []
        while pointer < timeseries.size:
            slice_ts = timeseries[pointer:(pointer+window_size_frames-1)]
            pointer += int(0.02 * sample_rate)
            slices.append(Util.AudioFile(slice_ts, sample_rate))
        return slices

    @staticmethod
    def extract_features(audiofile):
        mfcc = librosa.feature.mfcc(audiofile.timeseries, audiofile.sample_rate)
        audiofile.mfcc = mfcc

    @staticmethod
    def audio_dist(a1, a2):
        return Util.mfcc_dist(a1.mfcc, a2.mfcc)

    @staticmethod
    def mfcc_dist(mfcc1, mfcc2):
        dist, cost, acc_cost, path = dtw(
                mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1)
                )
        return dist
