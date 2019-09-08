import librosa
from dataclasses import dataclass
import numpy as np
import soundfile as sf

class Util:
    @dataclass
    class AudioFile:
        timeseries: np.ndarray
        sample_rate: int

    @staticmethod
    def read_audio(path):
        timeseries, sample_rate = librosa.load(path)
        return Util.AudioFile(timeseries, sample_rate)

    @staticmethod
    def chop_audio(audiofile, window_size_ms):
        pointer = 0
        timeseries = audiofile.timeseries
        sample_rate = audiofile.sample_rate

        window_size_frames = int((window_size_ms / 1000) * sample_rate)
        slices = []
        while pointer < timeseries.size:
            slice_ts = timeseries[pointer:(pointer+window_size_frames-1)]
            pointer += window_size_frames
            slices.append(Util.AudioFile(slice_ts, sample_rate))
        return slices

    @staticmethod
    def save_audio(audiofile, path):
        sf.write(path, audiofile.timeseries, audiofile.sample_rate, format='wav')
