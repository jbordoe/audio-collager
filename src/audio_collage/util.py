import numpy as np

from typing import List

from .audio_segment import AudioSegment
from .collage_progress_state import CollageProgressState

class Util:
    class SampleRateMismatchError(Exception):
        pass

    @staticmethod
    def chop_audio(
        audio_segment: AudioSegment,
        window_size_ms: int,
        step_ms: int = None,
        step_factor: float = None,
        progress_callback: callable = None
    ):
        if step_ms is not None and step_factor is not None:
            raise ValueError("Cannot specify both step_ms and step_factor")

        timeseries = audio_segment.timeseries
        sample_rate = audio_segment.sample_rate

        window_size_frames = int((window_size_ms / 1000) * sample_rate)

        if step_factor:
            step_ms = window_size_ms * step_factor
        # TODO: warn if step_ms is too small or too large
        if step_ms is None:
            step_frames = window_size_frames
        else:
            step_frames = int((step_ms / 1000) * sample_rate)

        if progress_callback:
            state = CollageProgressState(
                CollageProgressState.Task.CHOPPING,
                starting=True,
                current_step=0,
                total_steps=timeseries.size,
                message=f"Chopping {window_size_ms}ms window"
            )
            progress_callback(state)
        slices = []
        start_pointer, end_pointer = 0, window_size_frames
        while start_pointer < timeseries.size:
            slice_ts = timeseries[start_pointer:end_pointer]
            slices.append(AudioSegment(
                slice_ts,
                sample_rate,
                offset_frames=start_pointer
            ))
            start_pointer += step_frames
            end_pointer += step_frames

            if progress_callback:
                state = CollageProgressState(
                    CollageProgressState.Task.CHOPPING,
                    current_step=start_pointer,
                )
                progress_callback(state)

            if end_pointer > timeseries.size:
                break

        if progress_callback:
            state = CollageProgressState(
                CollageProgressState.Task.CHOPPING,
                completed=True,
                current_step=timeseries.size,
            )
            progress_callback(state)
        return slices

    @staticmethod
    def concatenate_audio(
        audio_list: List[AudioSegment],
        declick_fn: str = None,
        declick_ms: int = 0,
        sample_rate: int = 44100
    ):
        if not audio_list:
            return AudioSegment(np.array([]), sample_rate=sample_rate)

        output_timeseries = np.array([])

        for i, snippet in enumerate(audio_list):
            if snippet.sample_rate != sample_rate:
                # TODO: maybe we can resample the snippets to the same sample rate?
                raise Util.SampleRateMismatchError(f"Sample rates must match. Got {snippet.sample_rate} and {sample_rate}")

            snippet_ts = snippet.timeseries

            if declick_ms and len(output_timeseries):
                overlap_frames = int((declick_ms * snippet.sample_rate) / 1000)
                # Apply fade out to the end of the previous snippet
                output_timeseries = Util.declick_out(
                    output_timeseries,
                    n_frames=overlap_frames,
                    declick_type=declick_fn
                )
                # Apply fade in to the start of the current snippet
                snippet_ts = Util.declick_in(
                    snippet_ts,
                    n_frames=overlap_frames,
                    declick_type=declick_fn
                )
                # mix start of current snippet with end of the previous
                output_timeseries[-overlap_frames:] += snippet_ts[:overlap_frames]
                # Concatenate the rest of the snippet
                output_timeseries = np.concatenate(
                    [output_timeseries, snippet_ts[overlap_frames:]]
                )
            else:
                 output_timeseries = np.concatenate([output_timeseries, snippet_ts])

        return AudioSegment(output_timeseries, sample_rate)

    @staticmethod
    def declick_in(
        timeseries: np.ndarray,
        n_frames: int,
        declick_type: str,
        in_place = False
    ):
        if declick_type == 'linear':
            vector = Util.__declick_in_vector_linear(n_frames)
        elif declick_type == 'sigmoid':
            vector = Util.__declick_in_vector_sigmoid(n_frames)
        else:
            raise ValueError(f'Invalid declick type: {declick_type}')

        if not in_place:
            declicked = np.copy(timeseries)
        else:
            declicked = timeseries

        declicked[:n_frames] *= vector
        return declicked

    @staticmethod
    def declick_out(
        timeseries: np.ndarray,
        n_frames: int,
        declick_type: str,
        in_place = False
    ):
        if declick_type == 'linear':
            vector = Util.__declick_out_vector_linear(n_frames)
        elif declick_type == 'sigmoid':
            vector = Util.__declick_out_vector_sigmoid(n_frames)
        else:
            raise ValueError(f'Invalid declick type: {declick_type}')

        if not in_place:
            declicked = np.copy(timeseries)
        else:
            declicked = timeseries

        declicked[-n_frames:] *= vector
        return declicked

    def __declick_in_vector_linear(n_frames: int):
        return np.linspace(0., 1., n_frames)

    def __declick_out_vector_linear(n_frames: int):
        return np.linspace(1., 0., n_frames)

    def __declick_in_vector_sigmoid(n_frames: int):
        lin = np.linspace(0., 1., n_frames)
        steepness = 15
        return 1 / (1 + np.exp((0.5-lin) * steepness))

    def __declick_out_vector_sigmoid(n_frames: int):
        return np.flip(Util.__declick_in_vector_sigmoid(n_frames))
