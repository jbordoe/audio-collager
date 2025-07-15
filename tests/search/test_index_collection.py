from audio_collage.search.index_collection import SearchIndexCollection
from audio_collage.audio_dist import AudioDist
from audio_collage.audio_segment import AudioSegment
from audio_collage.search.index import SearchIndex
from audio_collage.util import Util

import numpy as np
from typing import Tuple

def mock_search_index(query_segment: AudioSegment) -> Tuple[float, AudioSegment]:
    timeseries = query_segment.timeseries
    window_size = int(timeseries.size / query_segment.sample_rate * 1000)
    return window_size, query_segment

def test_init():
    """
    Test that the SearchIndexCollection can be initialized with a distance function.
    """
    index_collection = SearchIndexCollection(AudioDist.mfcc_dist)
    
    assert isinstance(index_collection, SearchIndexCollection)
    assert index_collection.distance_fn == AudioDist.mfcc_dist
    assert len(index_collection.indices) == 0

def test_add_index(mocker):
    """
    Test that an index can be added to the collection.
    """
    mocked_index_build = mocker.patch.object(SearchIndex, 'build')
  
    index_collection = SearchIndexCollection(AudioDist.mfcc_dist)
    audio_segments = [
        AudioSegment(timeseries=np.arange(0, 10), sample_rate=1000),
        AudioSegment(timeseries=np.arange(10, 20), sample_rate=1000) ,
    ]
    index_collection.add_index(audio_segments, window=1000)

    assert len(index_collection.indices) == 1
    assert mocked_index_build.call_count == 1

def test_find_best_match(mocker):
    """
    Test that the best match is found from the corresponding index.
    """
    mocker.patch.object(SearchIndex, 'build')
    mocker.patch.object(Util, 'extract_features')
    mocked_search = mocker.patch.object(SearchIndex, 'search', side_effect=mock_search_index)
    
    index_collection = SearchIndexCollection(AudioDist.mfcc_dist)
    for window in [1, 2, 3]:
        audio_segments = [
            AudioSegment(timeseries=np.arange(0, window), sample_rate=1000),
            AudioSegment(timeseries=np.arange(window, window * 2), sample_rate=1000),
        ]
        index_collection.add_index(audio_segments, window=window)

    query_1 = AudioSegment(timeseries=np.array([1]), sample_rate=1000)
    match_1, dist_1, window_1 = index_collection.find_best_match(query_1)

    assert isinstance(match_1, AudioSegment)
    assert dist_1 == 1
    assert isinstance(window_1, int)

    query_2 = AudioSegment(timeseries=np.array([2, 3]), sample_rate=1000)
    match_2, dist_2, window_2 = index_collection.find_best_match(query_2)
    
    assert isinstance(match_2, AudioSegment)
    assert dist_2 == 1
    assert isinstance(window_2, int)

    query_3 = AudioSegment(timeseries=np.array([3, 4, 5]), sample_rate=1000)
    match_3, dist_3, window_3 = index_collection.find_best_match(query_3)

    assert isinstance(match_3, AudioSegment)
    assert dist_3 == 1
    assert isinstance(window_3, int)
