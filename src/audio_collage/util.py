import numpy as np

import librosa
import math
from typing import List

from .audio_segment import AudioSegment

class Util:
    @staticmethod
    def chop_audio(audiofile: AudioSegment, window_size_ms: int):
        pointer = 0
        timeseries = audiofile.timeseries
        sample_rate = audiofile.sample_rate

        window_size_frames = int((window_size_ms / 1000) * sample_rate)
        slices = []
        while pointer < timeseries.size:
            slice_ts = timeseries[pointer:(pointer+window_size_frames-1)]
            pointer += max(50, window_size_frames // 4)
            slices.append(AudioSegment(
                slice_ts,
                sample_rate,
                offset_frames=pointer
            ))
        return slices

    @staticmethod
    def concatenate_audio(
        audio_list: List[AudioSegment],
        declick_fn: str = None,
        declick_ms: int = 0
    ):
        output_data = []
        sample_rate = None
        for i, snippet in enumerate(audio_list):
            if declick_fn:
                snippet = Util.declick(snippet, declick_fn, declick_ms)

            x = snippet.timeseries
            if declick_ms and output_data and i < len(list(audio_list))-1:
                overlap_frames = int((declick_ms * snippet.sample_rate) / 1000)
                overlap = np.add(output_data[-overlap_frames:], x[:overlap_frames])
                output_data = output_data[:-overlap_frames]
                sample_rate = snippet.sample_rate if not sample_rate else sample_rate
                x = np.concatenate([overlap, x[overlap_frames:]])

            output_data.extend(x)
        output_audio = AudioSegment(np.array(output_data), sample_rate)
        return output_audio

    @staticmethod
    def declick(
        audiofile: AudioSegment,
        dc_type: str,
        fade_ms: int
    ):
        x = audiofile.timeseries
        sr = audiofile.sample_rate

        fade_frames = (sr * fade_ms) / 1000
        frames = x.size

        declick_functions = {
            'sigmoid': Util.__declick_vector_sigmoid,
            'linear':  Util.__declick_vector_linear,
        }
        fn = declick_functions[dc_type](frames, fade_frames)
        declicked = x * fn

        return AudioSegment(declicked, sr, offset_frames=audiofile.offset_frames)

    def __declick_vector_linear(n_frames: int, fade_frames: int):
        return [
            min(
                min(i/fade_frames, 1),
                min((n_frames-i)/fade_frames, 1)
            )
            for i in np.arange(0, n_frames, 1)
        ]
   
    def __declick_vector_sigmoid(n_frames: int, fade_frames: int):
        return [
            min(
                1/(1+math.exp((0.5-(i/fade_frames))*15)),
                1/(1+math.exp((0.5-((n_frames-i)/fade_frames))*15))
            )
            for i in np.arange(0, n_frames, 1)
        ]
   
    @staticmethod
    def extract_features(audiofile: AudioSegment):
        audiofile.mfcc = librosa.feature.mfcc(
                y=audiofile.timeseries,
                sr=audiofile.sample_rate,
                n_fft = min(2048, len(audiofile.timeseries))
        )
        audiofile.mfcc_mean = np.mean(audiofile.mfcc, axis=1)
#        audiofile.chroma_stft = librosa.feature.chroma_stft(
#                y=audiofile.timeseries,
#                sr=audiofile.sample_rate,
#                n_fft = min(2048, len(audiofile.timeseries))
#        )

