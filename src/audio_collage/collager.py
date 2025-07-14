from rich.progress import Progress, track
from typing import Dict, List, Tuple

import os
import pickle
from vptree import VPTree

from .audio_file import AudioFile
from .util import Util

CACHE_DIR = '.cache'

class Collager:
    def __init__(
        self,
        target_audio: AudioFile,
        sample_audio: AudioFile,
        distance_fn: callable = Util.mean_mfcc_dist
    ):
        self.source: AudioFile = sample_audio
        self.target: AudioFile = target_audio
        self.indices: Dict[int, VPTree] = {}
        self.distance_fn = distance_fn

    def collage(self, windows: List[int] = [200], overlap_ms: int = 0) -> List[AudioFile]:
        self._chop(windows)
        selected_snippets: List[AudioFile] = []

        Util.extract_features(self.target)
        target_sr = self.target.sample_rate

        pointer: int = 0
        with Progress() as progress:
            task = progress.add_task('[cyan]Selecting samples...', total=100)
            while pointer < self.target.timeseries.size:
                pct_complete = (pointer / self.target.timeseries.size) * 100 if pointer else 0
                progress.update(task, completed=pct_complete)

                best_snippet: AudioFile = None
                best_snippet_dist: float = float('inf')
                best_snippet_window: int = 0
                for window in windows:
                    window_size_frames = int((window / 1000) * target_sr)
                    target_chunk = AudioFile(
                            self.target.timeseries[pointer:pointer + window_size_frames - 1],
                            target_sr,
                            )
                    Util.extract_features(target_chunk)

                    nearest_dist, nearest = self._search(target_chunk, window)
                    if nearest_dist < best_snippet_dist:
                        best_snippet_dist = nearest_dist
                        best_snippet = nearest
                        best_snippet_window = window_size_frames

                selected_snippets.append(best_snippet)
                pointer += best_snippet_window - int((overlap_ms /1000) * target_sr)
            progress.update(task, completed=100)

        return selected_snippets

    def _chop(self, windows: List[int]) -> None:
        with Progress() as progress:
            task = progress.add_task(
                description="[cyan]Chopping and analysing sample audio...",
                total=self.source.n_samples() * len(windows)
            )
            hash: str = self.source.hash()
            for window in windows:
                if self._cache_load(hash, window):
                    progress.update(task, advance = self.source.n_samples())
                    continue

                sample_group: List[AudioFile] = Util.chop_audio(self.source, window)

                for s in sample_group:
                    Util.extract_features(s)
                    progress.update(task, advance=int((window / 1000) * self.source.sample_rate))

                self._index(sample_group, window, hash)

    def _search(self, query_audio: AudioFile, key: int) -> Tuple[float, AudioFile]:
        index = self.indices[key]
        nearest_dist, nearest = index.get_nearest_neighbor(query_audio)
        return nearest_dist, nearest

    def _index(self, samples: List[AudioFile], key: int, hash: str) -> None:
        tree = VPTree(samples, self.distance_fn)
        self.indices[key] = tree
        self._cache_vptree(hash, key)
    
    def _cache_path(self, hash: str, key: int) -> str:
        return os.path.join(
            CACHE_DIR,
            f"{hash}.{key}.{self.distance_fn.__name__}.vptree"
        )

    def _cache_vptree(self, hash: str, key: int) -> None:
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

        cache_path: str = self._cache_path(hash, key)
        with open(cache_path, 'wb') as f:
            pickle.dump(self.indices[key], f)

    def _clear_cache(self, hash: str, key: int) -> None:
        cache_path: str = self._cache_path(hash, key)
        if os.path.exists(cache_path):
            os.remove(cache_path)

    def _cache_load(self, hash: str, key: int) -> bool:
        if not os.path.exists(CACHE_DIR):
            return False

        cache_path: str = self._cache_path(hash, key)
        try:
            with open(cache_path, 'rb') as f:
                index: VPTree = pickle.load(f)
                if index is None or not isinstance(index, VPTree):
                    raise EOFError
            
                self.indices[key] = index
                return True
        except FileNotFoundError:
            return False
        except EOFError:
            print(f'Error loading cache, removing file')
            self._clear_cache(hash, key)
            return False
