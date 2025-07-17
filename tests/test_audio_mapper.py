from audio_collage.audio_dist import AudioDist
from audio_collage.audio_mapper import AudioMapper
from audio_collage.audio_segment import AudioSegment
from audio_collage.collager import CollagerConfig
from audio_collage.search.index_collection import SearchIndexCollection
from audio_collage.util import Util

import numpy as np

def test_init():
    """
    Test initializing an AudioMapper object
    """
    source = AudioSegment(None, None)
    target = AudioSegment(None, None)
    mapper = AudioMapper(source, target)

    assert mapper.source == source
    assert mapper.target == target
    assert isinstance(mapper.indices, SearchIndexCollection)
    assert mapper.distance_fn == AudioDist.mean_mfcc_dist

def test_map_audio(mocker):
    """
    Test that the audio is mapped correctly.
    """
    mocker.patch.object(SearchIndexCollection, 'find_best_match', return_value=(111, 22, 3))
    chop_fn = mocker.spy(Util, 'chop_audio')
    
    config = CollagerConfig(
        step_ms=100,
        windows=[100, 200]
    )
    source = AudioSegment(timeseries=np.arange(0, 10), sample_rate=1000)
    target = AudioSegment(timeseries=np.arange(0, 10), sample_rate=1000)
    mapper = AudioMapper(
        source,
        target,
        config=config
    )

    selected_snippets = mapper.map_audio()

    assert len(selected_snippets) == 4
    for window in config.windows:
        chop_fn.assert_any_call(
            source,
            window,
            step_ms=100,
            step_factor=None
        )
