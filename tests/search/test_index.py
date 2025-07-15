from audio_collage.search.index import SearchIndex
from audio_collage.audio_dist import AudioDist
from audio_collage.audio_segment import AudioSegment
from audio_collage.util import Util

import numpy as np
from typing import Tuple
from vptree import VPTree

def test_init():
    """
    Test that the SearchIndex can be initialized with a distance function.
    """
    index = SearchIndex(window_size=1000, distance_fn=AudioDist.mfcc_dist)
    
    assert isinstance(index, SearchIndex)
    assert index.distance_fn == AudioDist.mfcc_dist
    assert index.tree is None

def test_build():
    """
    Test that the VP-tree is built from the audio segments.
    """
    
    index = SearchIndex(window_size=1000, distance_fn=AudioDist.mfcc_dist)
    audio_segments = [
        AudioSegment(timeseries=np.arange(0, 10, dtype=float), sample_rate=1000),
        AudioSegment(timeseries=np.arange(10, 20, dtype=float), sample_rate=1000),
    ]
    index.build(audio_segments)

    assert isinstance(index.tree, VPTree)

def test_search():
    """
    Test that the nearest neighbor is found from the VP-tree.
    """
    index = SearchIndex(window_size=1000, distance_fn=AudioDist.mfcc_dist)
    audio_segments = [
        AudioSegment(timeseries=np.arange(0, 10, dtype=float), sample_rate=1000),
        AudioSegment(timeseries=np.arange(10, 20, dtype=float), sample_rate=1000),
    ]
    index.build(audio_segments)

    query_segment = audio_segments[0]
    Util.extract_features(query_segment)

    nearest_dist, nearest = index.search(query_segment)

    assert isinstance(nearest, AudioSegment)
    assert nearest_dist == AudioDist.mfcc_dist(query_segment, audio_segments[0])
