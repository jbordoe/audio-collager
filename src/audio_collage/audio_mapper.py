from rich.progress import Progress
from typing import List, Tuple


from .audio_dist import AudioDist
from .audio_segment import AudioSegment
from .collager_config import CollagerConfig
from .search.index_collection import SearchIndexCollection
from .util import Util

CACHE_DIR = '.cache'

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

        with Progress() as progress:
            task = progress.add_task('[cyan]Selecting samples...', total=100)
            while pointer < n_frames:
                pct_complete = (pointer / n_frames) * 100
                progress.update(task, completed=pct_complete)

                target_ts = self.target.timeseries[pointer:]
                target_chunk = AudioSegment(target_ts, target_sr)

                best_snippet, best_dist, best_window_ms = self._search(target_chunk)

                if best_snippet:
                    selected_snippets.append(best_snippet)

                if best_snippet is None:
                    break

                pointer += best_window_ms - int((self.config.declick_ms / 1000) * target_sr)
                progress.update(task, advance=int((best_window_ms / 1000) * target_sr))
            progress.update(task, completed=100)

        return selected_snippets

    def _chop(self) -> None:
        windows = self.config.windows
        windows = [i + self.config.declick_ms for i in windows]

        with Progress() as progress:
            task = progress.add_task(
                description="[cyan]Chopping and analysing sample audio...",
                total=self.source.n_samples() * len(windows)
            )
            for window in windows:
                sample_group: List[AudioSegment] = Util.chop_audio(
                    self.source,
                    window,
                    step_ms=self.config.step_ms,
                    step_factor=self.config.step_factor
                )

                self._index(sample_group, window)
                progress.update(
                    task,
                    advance=int((window / 1000) * self.source.sample_rate) * len(sample_group)
                )
            progress.update(task, completed=100)

    def _search(self, query_audio: AudioSegment) -> Tuple[float, AudioSegment]:
        return self.indices.find_best_match(query_audio)

    def _index(self, samples: List[AudioSegment], window: int) -> None:
        self.indices.add_index(samples, window)
