import numpy as np

import math
from typing import List

from .audio_segment import AudioSegment

class Util:
    @staticmethod
    def chop_audio(audio_segment: AudioSegment, window_size_ms: int):
        pointer = 0
        timeseries = audio_segment.timeseries
        sample_rate = audio_segment.sample_rate

        window_size_frames = int((window_size_ms / 1000) * sample_rate)
        slices = []
        while pointer < timeseries.size:
            slice_ts = timeseries[pointer:(pointer+window_size_frames)]
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
        audio_segment: AudioSegment,
        dc_type: str,
        fade_ms: int
    ):
        x = audio_segment.timeseries
        sr = audio_segment.sample_rate

        fade_frames = (sr * fade_ms) / 1000
        frames = x.size

        declick_functions = {
            'sigmoid': Util.__declick_vector_sigmoid,
            'linear':  Util.__declick_vector_linear,
        }
        vector = declick_functions[dc_type](frames, fade_frames)
        declicked = x * vector

        return AudioSegment(declicked, sr, offset_frames=audio_segment.offset_frames)

    def __declick_vector_linear(n_frames: int, fade_frames: int):
        return [
            min(
                min(i/fade_frames, 1),
                min((n_frames-(i+1))/fade_frames, 1)
            )
            for i in np.arange(0, n_frames, 1)
        ]

   
    def __declick_vector_sigmoid(n_frames: int, fade_frames: int):
        return [
            min(
                1/(1+math.exp((0.5-(i/fade_frames))*15)),
                1/(1+math.exp((0.5-((n_frames-(i+1))/fade_frames))*15))
            )
            for i in np.arange(0, n_frames, 1)
        ]

