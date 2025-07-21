from typing import Callable, List, Tuple


from .audio_dist import AudioDist
from .audio_segment import AudioSegment
from .collager_config import CollagerConfig
from .collage_progress_state import CollageProgressState
from .search.index_collection import SearchIndexCollection
from .util import Util

CACHE_DIR = '.cache'

class AudioMapper:
    def __init__(
        self,
        sample_audio: AudioSegment,
        target_audio: AudioSegment,
        distance_fn: Callable = AudioDist.mean_mfcc_dist,
        config: CollagerConfig = CollagerConfig()
    ):
        self.source: AudioSegment = sample_audio
        self.target: AudioSegment = target_audio
        self.indices: SearchIndexCollection = SearchIndexCollection(distance_fn)
        self.distance_fn = distance_fn
        self.config = config

    def map_audio(self) -> List[AudioSegment]:
        """
        Maps the target audio to the source audio using the specified windows.
        
        Returns:
            List[AudioSegment]: List of selected snippets.
        """
        self._chop()
        selected_snippets: List[AudioSegment] = []

        target_sr: int = self.target.sample_rate
        n_frames: int = self.target.timeseries.size
        pointer: int = 0

        if self.config.progress_callback:
            self.config.progress_callback(CollageProgressState(
                CollageProgressState.Task.SELECTING,
                starting=True,
                current_step=0,
                total_steps=n_frames,
                message="Selecting samples"
            ))
        while pointer < n_frames:
            target_ts = self.target.timeseries[pointer:]
            target_chunk = AudioSegment(target_ts, target_sr)

            best_snippet, best_dist, best_n_frames = self._search(target_chunk)

            if best_snippet:
                selected_snippets.append(best_snippet)
            else:
                break

            advance = best_n_frames - int((self.config.declick_ms / 1000) * target_sr)
            pointer += advance
            if self.config.progress_callback:
                self.config.progress_callback(CollageProgressState(
                    CollageProgressState.Task.SELECTING,
                    advance=advance,
                ))


        if self.config.progress_callback:
            self.config.progress_callback(CollageProgressState(
                CollageProgressState.Task.SELECTING,
                completed=True,
                current_step=sum([snip.timeseries.size for snip in selected_snippets]),
            ))

        return selected_snippets

    def _chop(self) -> None:
        windows = self.config.windows
        windows = [i + self.config.declick_ms for i in windows]

        if self.config.progress_callback:
            self.config.progress_callback(CollageProgressState(
                CollageProgressState.Task.CHOPPING,
                starting=True,
                current_step=0,
                total_steps=len(windows),
                message=f"Chopping {len(windows)} windows"
            ))

        for window in windows:
            sample_group: List[AudioSegment] = Util.chop_audio(
                self.source,
                window,
                step_ms=self.config.step_ms,
                step_factor=self.config.step_factor
            )

            self._index(sample_group, window)
            if self.config.progress_callback:
                self.config.progress_callback(CollageProgressState(
                    CollageProgressState.Task.CHOPPING,
                    current_step=len(sample_group),
                ))
        if self.config.progress_callback:
            self.config.progress_callback(CollageProgressState(
                CollageProgressState.Task.CHOPPING,
                completed=True,
                current_step=len(windows),
            ))

    def _search(self, query_audio: AudioSegment) -> Tuple[AudioSegment, float, int]:
        return self.indices.find_best_match(query_audio)

    def _index(self, samples: List[AudioSegment], window: int) -> None:
        self.indices.add_index(samples, window)
