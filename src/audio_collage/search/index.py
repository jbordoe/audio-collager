import hashlib
import os
import pickle
from typing import List, Tuple, Callable
from vptree import VPTree

from ..audio_segment import AudioSegment

CACHE_DIR = '.cache'

class SearchIndex:
    """
    Manages a single VP-tree search index for a specific window size and distance function,
    including building, searching, and caching.
    """
    def __init__(self, window_size: int, distance_fn: Callable[[AudioSegment, AudioSegment], float]):
        self.window_size = window_size
        self.distance_fn = distance_fn
        self.tree: VPTree = None

    def build(self, audio_segments: List[AudioSegment]) -> None:
        """
        Builds the VP-tree index, loading from cache if available, otherwise
        building from scratch and caching the result.
        """
        hash = self.audio_segments_hash(audio_segments)
        if self._load_from_cache(hash):
            return

        self.tree = VPTree(audio_segments, self.distance_fn)
        self._save_to_cache(hash)

    def search(self, query_segment: AudioSegment) -> Tuple[float, AudioSegment]:
        """
        Searches the VP-tree for the nearest neighbor to the query segment.
        """
        if not self.tree:
            raise RuntimeError("SearchIndex has not been built yet.")
        
        return self.tree.get_nearest_neighbor(query_segment)

    def _get_cache_path(self, source_hash: str) -> str:
        """
        Determines the file path for the cached index.
        """
        return os.path.join(
            CACHE_DIR,
            f"{source_hash}.{self.window_size}.{self.distance_fn.__name__}.vptree"
        )

    def _load_from_cache(self, source_hash: str) -> bool:
        """
        Loads the VP-tree from the cache if it exists and is valid.
        """
        cache_path = self._get_cache_path(source_hash)
        if not os.path.exists(cache_path):
            return False
        
        try:
            with open(cache_path, 'rb') as f:
                tree = pickle.load(f)
                if not isinstance(tree, VPTree):
                    raise TypeError("Cached object is not a VPTree")
                self.tree = tree
                return True
        except (pickle.UnpicklingError, EOFError, TypeError) as e:
            print(f"Warning: Could not load cache file {cache_path}. It will be rebuilt. Error: {e}")
            os.remove(cache_path)
            return False

    def _save_to_cache(self, source_hash: str) -> None:
        """
        Saves the built VP-tree to the cache.
        """
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        
        cache_path = self._get_cache_path(source_hash)
        with open(cache_path, 'wb') as f:
            pickle.dump(self.tree, f)

    def audio_segments_hash(self, audio_segments: List[AudioSegment]) -> str:
        """
        Generates a hash for the given audio segments.
        """
        return hashlib.md5(pickle.dumps(audio_segments)).hexdigest()
