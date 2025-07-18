from typing import List, Tuple
from joblib import Parallel, delayed, parallel_backend
from joblib.externals.loky import set_loky_pickler

from .audio_dist import AudioDist
from .audio_segment import AudioSegment
from .collager_config import CollagerConfig
from .collage_progress_state import CollageProgressState
from .search.index_collection import SearchIndexCollection
from .util import Util

CACHE_DIR = '.cache'

# NOTE: this is a temporary solution to the problem of not being able to pickle
# because of the use of bound methods for distance functions
def _map_subtask(
    target: AudioSegment,
    indices: SearchIndexCollection,
    declick_ms: int,
    progress_callback: callable,
) -> List[AudioSegment]:
    selected_snippets: List[AudioSegment] = []
    target_sr: int = target.sample_rate
    n_frames: int = target.timeseries.size
    pointer: int = 0
    while pointer < n_frames:
        target_ts = target.timeseries[pointer:]
        target_chunk = AudioSegment(target_ts, target_sr)

        best_snippet, best_dist, best_n_frames = indices.find_best_match(target_chunk)

        if best_snippet:
            selected_snippets.append(best_snippet)
        else:
            break

        advance = best_n_frames - int((declick_ms / 1000) * target_sr)
        pointer += advance
        if progress_callback:
            progress_callback(CollageProgressState(
                CollageProgressState.Task.SELECTING,
                advance=advance,
            ))
    return selected_snippets

class AudioMapper:
    def __init__(
        self,
        sample_audio: AudioSegment,
        target_audio: AudioSegment,
        distance_fn: callable = AudioDist.mean_mfcc_dist,
        config: CollagerConfig = CollagerConfig()
    ):
        self.source: AudioSegment = sample_audio
        self.target: AudioSegment = target_audio
        self.indices: SearchIndexCollection = SearchIndexCollection(distance_fn)
        self.config = config

    def map_audio(self) -> List[AudioSegment]:
        """
        Maps the target audio to the source audio using the specified windows.

        Returns:
            List[AudioSegment]: List of selected snippets.
        """
        self._chop()
        selected_snippets: List[AudioSegment] = []

        # TODO: move to config
        n_jobs: int = 8
        target_chunks = self.target.split(n_jobs)

        n_frames: int = self.target.timeseries.size

        if self.config.progress_callback:
            self.config.progress_callback(CollageProgressState(
                CollageProgressState.Task.SELECTING,
                starting=True,
                current_step=0,
                total_steps=n_frames,
                message="Selecting samples"
            ))
        set_loky_pickler('dill')
        with parallel_backend('threading', n_jobs=n_jobs):
            results = Parallel()(
                delayed(_map_subtask)(
                    target_chunk,
                    self.indices,
                    self.config.declick_ms,
                    self.config.progress_callback
                )
                for target_chunk in target_chunks
            )
        # flatten list of lists
        selected_snippets = [snip for snips in results for snip in snips]

        if self.config.progress_callback:
            self.config.progress_callback(CollageProgressState(
                CollageProgressState.Task.SELECTING,
                completed=True,
                current_step=n_frames,
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

    def _search(self, query_audio: AudioSegment) -> Tuple[float, AudioSegment]:
        return self.indices.find_best_match(query_audio)

    def _index(self, samples: List[AudioSegment], window: int) -> None:
        self.indices.add_index(samples, window)
