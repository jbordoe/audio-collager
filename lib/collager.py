from rich.progress import Progress, track

import vptree

from lib.util import Util

class Collager:
    def __init__(self, target_audio, sample_audio):
        self.source = sample_audio
        self.target = target_audio
        self.indices = {}
        pass

    def collage(self, windows=[200], overlap_ms=0): 
        samples = {}

        with Progress() as progress:
            task = progress.add_task(
                description="[cyan]Chopping and analysing sample audio...",
                total=self.source.n_samples() * len(windows)
            )
            for window in windows:
                sample_group = Util.chop_audio(self.source, window)

                for s in sample_group:
                    Util.extract_features(s)
                    progress.update(task, advance=int((window / 1000) * self.source.sample_rate))

                self._index(sample_group, window)

        selected_snippets = []

        Util.extract_features(self.target)
        target_sr = self.target.sample_rate

        pointer = 0
        with Progress() as progress:
            task = progress.add_task('[cyan]Selecting samples...', total=100)
            while pointer < self.target.timeseries.size:
                pct_complete = (pointer / self.target.timeseries.size) * 100 if pointer else 0
                progress.update(task, completed=pct_complete)

                best_snippet = None
                best_snippet_dist = 999999
                best_snippet_window = None
                for window in windows:
                    window_size_frames = int((window / 1000) * target_sr)
                    target_chunk = Util.AudioFile(
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

    def _search(self, query_audio, key):
        index = self.indices[key]
        nearest_dist, nearest = index.get_nearest_neighbor(query_audio)
        return nearest_dist, nearest

    def _index(self, samples, key):
        tree = vptree.VPTree(samples, Util.mfcc_dist)
        self.indices[key] = tree
