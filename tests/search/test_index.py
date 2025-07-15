from audio_collage.search.index import SearchIndex
from audio_collage.audio_dist import AudioDist
from audio_collage.audio_segment import AudioSegment

import numpy as np
import pytest
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

def test_build(mocker):
    """
    Test that the VP-tree is built from the audio segments.
    """
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('pickle.dump')
    mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('os.makedirs')

    index = SearchIndex(window_size=1000, distance_fn=AudioDist.mfcc_dist)
    segments = [
        AudioSegment(timeseries=np.arange(0, 10, dtype=float), sample_rate=1000),
        AudioSegment(timeseries=np.arange(10, 20, dtype=float), sample_rate=1000) ,
    ]

    index.build(segments)

    assert isinstance(index.tree, VPTree)

def test_build_loads_from_cache_if_exists(mocker):
    """
    1. Tests that the VPTree is loaded from an existing cache. 
    """
    mocker.patch('os.path.exists', return_value=True)
    mock_vptree = mocker.MagicMock(spec=VPTree)
    mock_pickle_load = mocker.patch('pickle.load', return_value=mock_vptree)
    mocker.patch('builtins.open', mocker.mock_open(read_data=b'fake_data'))
    
    # Mock the VPTree constructor to ensure it's NOT called
    mock_vptree_constructor = mocker.patch('vptree.VPTree')

    index = SearchIndex(window_size=1000, distance_fn=AudioDist.mfcc_dist)
    segments = [
        AudioSegment(timeseries=np.arange(0, 10, dtype=float), sample_rate=1000),
        AudioSegment(timeseries=np.arange(10, 20, dtype=float), sample_rate=1000) ,
    ]

    index.build(segments)

    mock_pickle_load.assert_called_once()
    mock_vptree_constructor.assert_not_called()
    assert index.tree is mock_vptree

def test_clears_cache_if_format_invalid(mocker):
    """
    Test that the cache is cleared if loading fails.
    """
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('builtins.open', mocker.mock_open(read_data=b'fake_data'))
    mocker.patch('pickle.load', side_effect=EOFError)
    mocked_cache_remove = mocker.patch('os.remove')

    index = SearchIndex(window_size=1000, distance_fn=AudioDist.mfcc_dist)
    segments = [
        AudioSegment(timeseries=np.arange(10, dtype=float), sample_rate=1000),
        AudioSegment(timeseries=np.arange(20, dtype=float), sample_rate=1000),
    ]

    index.build(segments)

    mocked_cache_remove.assert_called_once()

def test_clears_cache_if_not_vptree(mocker):
    """
    Test that the cache is cleared if the loaded object is not a VPTree.
    """
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('builtins.open', mocker.mock_open(read_data=b'fake_data'))
    mocker.patch('pickle.load', return_value=123)
    mocked_cache_remove = mocker.patch('os.remove')

    index = SearchIndex(window_size=1000, distance_fn=AudioDist.mfcc_dist)
    segments = [
        AudioSegment(timeseries=np.arange(10, dtype=float), sample_rate=1000),
        AudioSegment(timeseries=np.arange(20, dtype=float), sample_rate=1000),
    ]

    index.build(segments)

    mocked_cache_remove.assert_called_once()

def test_build_creates_new_index_if_no_cache(mocker):
    """
    2. Tests that a new VPTree is built if no cache is found.
    """
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('pickle.dump')
    mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('os.makedirs') # Prevent trying to create .cache dir

    # Mock the VPTree constructor to ensure it's called
    mock_vptree = mocker.MagicMock(spec=VPTree)
    mock_vptree_constructor = mocker.patch('vptree.VPTree', return_value=mock_vptree)

    index = SearchIndex(window_size=1000, distance_fn=AudioDist.mfcc_dist)
    segments = [
        AudioSegment(timeseries=np.arange(0, 10, dtype=float), sample_rate=1000),
        AudioSegment(timeseries=np.arange(10, 20, dtype=float), sample_rate=1000) ,
    ]

    index.build(segments)

    mock_vptree_constructor.assert_called_once()
    assert isinstance(index.tree, VPTree)

def test_build_writes_to_cache_after_creation(mocker):
    """
    3. Tests that a cache file is written after a new VPTree is built.
    """
    mocker.patch('os.path.exists', return_value=False)
    mock_pickle_dump = mocker.patch('pickle.dump')
    mock_open = mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('os.makedirs')

    index = SearchIndex(window_size=1000, distance_fn=AudioDist.mfcc_dist)
    segments = [
        AudioSegment(timeseries=np.arange(0, 10, dtype=float), sample_rate=1000),
        AudioSegment(timeseries=np.arange(10, 20, dtype=float), sample_rate=1000) ,
    ]

    index.build(segments)

    mock_open.assert_called_once_with(mocker.ANY, 'wb') # Check it was opened for writing
    mock_pickle_dump.assert_called_once() # Check that we tried to save
    assert mock_pickle_dump.call_args[0][0] is index.tree # Check we saved the correct object

def test_search(mocker):
    """
    Test that the nearest neighbor is found from the VP-tree.
    """
    mocker.patch('os.path.exists', return_value=False)
    mocker.patch('pickle.dump')
    mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('os.makedirs')

    index = SearchIndex(window_size=1000, distance_fn=AudioDist.mfcc_dist)
    segments = [
        AudioSegment(timeseries=np.arange(0, 10, dtype=float), sample_rate=1000),
        AudioSegment(timeseries=np.arange(10, 20, dtype=float), sample_rate=1000) ,
    ]

    index.build(segments)

    query_segment = AudioSegment(timeseries=np.arange(0, 1000, dtype=float), sample_rate=1000)

    nearest_dist, nearest = index.search(query_segment)

    assert isinstance(nearest, AudioSegment)
    assert nearest_dist is not None

def test_search_raises_error_if_no_tree(mocker):
    """
    Test that an error is raised if the tree is not built.
    """
    index = SearchIndex(window_size=1000, distance_fn=AudioDist.mfcc_dist)

    with pytest.raises(RuntimeError):
        query_segment = AudioSegment(
            timeseries=np.arange(10, dtype=float),
            sample_rate=1000
        )
        index.search(query_segment)
