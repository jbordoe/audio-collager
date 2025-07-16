from typing import Dict, List, Callable, Tuple
import numpy as np

from ..audio_segment import AudioSegment
from .index import SearchIndex

class SearchIndexCollection:
    """
    Manages a collection of SearchIndex objects, one for each specified window size.
    """
    def __init__(
        self,
        distance_fn: Callable[[AudioSegment, AudioSegment], float]
    ):
        self.distance_fn = distance_fn
        AudioSegment(timeseries=np.arange(10), sample_rate=1000),
        self.indices: Dict[int, SearchIndex] = {}

    def add_index(
        self,
        audio_segments: List[AudioSegment],
        window: int,
    ) -> None:
        """
        Initializes and builds all the search indices for the specified window sizes.
        """
        index = SearchIndex(window, self.distance_fn)
        index.build(audio_segments)
        self.indices[window] = index

    def find_best_match(
        self,
        query_segment: AudioSegment,
    ) -> Tuple[float, AudioSegment, int]:
        """
        Searches across all indices to find the single best matching segment.

        Returns:
            A tuple containing:
                - The best matching distance
                - Distance between the query segment and the best match
                - Window size of the best matching segment
        """
        best_overall_snippet: AudioSegment = None
        best_overall_dist: float = float('inf')
        best_overall_window: int = 0

        target_sr = query_segment.sample_rate
            
        for window_size, index in self.indices.items():
            window_size_frames = int((window_size / 1000) * target_sr)
            
            # Ensure we don't read past the end of the timeseries
            if len(query_segment.timeseries) < window_size_frames:
                continue

            target_chunk = AudioSegment(
                query_segment.timeseries[:window_size_frames],
                target_sr,
            )
            dist, snippet = index.search(target_chunk)
            normalized_dist = dist / window_size_frames

            if normalized_dist < best_overall_dist:
                best_overall_dist = normalized_dist
                best_overall_snippet = snippet
                best_overall_window = window_size_frames

        return best_overall_snippet, best_overall_dist, best_overall_window
