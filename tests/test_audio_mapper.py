from audio_collage.audio_dist import AudioDist
from audio_collage.audio_mapper import AudioMapper
from audio_collage.audio_segment import AudioSegment

def test_init():
    """
    Test initializing an AudioMapper object
    """
    source = AudioSegment(None, None)
    target = AudioSegment(None, None)
    mapper = AudioMapper(source, target)

    assert mapper.source == source
    assert mapper.target == target
    assert mapper.indices == {}
    assert mapper.distance_fn == AudioDist.mean_mfcc_dist
