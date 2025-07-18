from .util import Util
from .audio_dist import AudioDist
from .audio_mapper import AudioMapper
from .audio_segment import AudioSegment
from .collager_config import CollagerConfig

from typing import Dict, Callable

class Collager:
    @staticmethod
    def create_collage(
        target_audio: AudioSegment,
        sample_audio: AudioSegment,
        config: CollagerConfig
    ) -> AudioSegment:
        """
        This is the core logic for creating a collage.
        """
        declick_fn = config.declick_fn
        declick_ms = config.declick_ms
        distance_fn = config.distance_fn

        default_dc_ms = {
            'sigmoid': 20,
            'linear': 70,
        }
        if declick_fn:
            declick_ms = declick_ms or default_dc_ms[declick_fn]
        else:
            declick_ms = 0

        dist_fn_map: Dict[str, Callable[[AudioSegment, AudioSegment], float]] = {
            'mfcc': AudioDist.mfcc_dist,
            'fast_mfcc': AudioDist.fast_mfcc_dist,
            'mean_mfcc': AudioDist.mean_mfcc_dist,
            'mfcc_cosine': AudioDist.mfcc_cosine_dist,
        }
        selected_distance_fn = dist_fn_map.get(distance_fn.value)
        if not selected_distance_fn:
            raise ValueError(f'Invalid distance function: {distance_fn}')

        mapper = AudioMapper(
            sample_audio,
            target_audio,
            distance_fn=selected_distance_fn,
            config=config
        )

        selected_snippets = mapper.map_audio()

        output_audio = Util.concatenate_audio(
            selected_snippets,
            declick_fn=declick_fn,
            declick_ms=declick_ms,
            sample_rate=sample_audio.sample_rate,
            progress_callback=config.progress_callback
        )

        return output_audio

