from audio_collage.util import Util
from audio_collage.audio_segment import AudioSegment
import numpy as np

def test_chop_audio():
    """
    Test chopping an audio segment into smaller segments
    """
    audio_segment = AudioSegment(
        np.arange(100),
        sample_rate=100
    )
    chopped = Util.chop_audio(audio_segment, window_size_ms=500)
    assert len(chopped) == 2
    assert np.array_equal(chopped[0].timeseries, np.arange(50))
    assert np.array_equal(chopped[1].timeseries, np.arange(50, 100))

def test_concatenate_audio():
    """
    Test concatenating audio segments
    """
    audio_segment_1 = AudioSegment(
        np.arange(50),
        sample_rate=100
    )
    audio_segment_2 = AudioSegment(
        np.arange(50, 100),
        sample_rate=100
    )
    concatenated = Util.concatenate_audio([audio_segment_1, audio_segment_2])
    expected_timeseries = np.arange(100)
    assert np.array_equal(concatenated.timeseries, expected_timeseries)

def test_linear_declick():
    """
    Test linear declicking
    """
    audio_segment = AudioSegment(
        np.ones(10) * 2,
        sample_rate=10
    )
    declicked = Util.declick(audio_segment, 'linear', fade_ms=400)
    expected_timeseries = np.array([0., 0.5, 1., 1.5, 2., 2. ,1.5, 1., 0.5, 0.])
    assert np.array_equal(declicked.timeseries, expected_timeseries)

def test_sigmoid_declick():
    """
    Test sigmoid declicking
    """
    audio_segment = AudioSegment(
        np.ones(10) * 2,
        sample_rate=10
    )
    declicked = Util.declick(audio_segment, 'sigmoid', fade_ms=400)
    expected_timeseries = np.array([0.001, 0.046, 1.0, 1.954, 1.999, 1.999, 1.954, 1.0, 0.045, 0.001])
    # assert result is close to expected
    assert np.allclose(declicked.timeseries, expected_timeseries, atol=0.1)
