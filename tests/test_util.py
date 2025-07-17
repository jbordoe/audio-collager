from audio_collage.util import Util
from audio_collage.audio_segment import AudioSegment
import numpy as np
import pytest

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

def test_chop_audio_with_fixed_step():
    """
    Test chopping audio segments with a step size
    """
    audio_segment = AudioSegment(
        np.arange(100),
        sample_rate=100
    )
    chopped = Util.chop_audio(audio_segment, window_size_ms=500, step_ms=250)
    assert len(chopped) == 3
    assert np.array_equal(chopped[0].timeseries, np.arange(50))
    assert np.array_equal(chopped[1].timeseries, np.arange(25, 75))
    assert np.array_equal(chopped[2].timeseries, np.arange(50, 100))

def test_concatenate_audio_without_declick():
    """
    Test concatenating audio segments without declicking
    """
    sr = 44100
    n_segments = 4
    ts_list = [np.random.rand(sr * 1) for _ in range(n_segments)] # 1 second of audio
    segments = [AudioSegment(timeseries, sr) for timeseries in ts_list]
    
    result_segment = Util.concatenate_audio(segments)
    
    expected_length = sum([len(segment.timeseries) for segment in segments])
    actual_length = len(result_segment.timeseries)
    
    assert actual_length == expected_length, \
        f"Concatenated audio has incorrect length! Expected {expected_length}, but got {actual_length}"

def test_concatenate_audio_with_declick():
    """
    Test concatenating audio segments
    """
    sr = 44100
    n_segments = 4
    ts_list = [np.random.rand(sr * 1) for _ in range(n_segments)] # 1 second of audio
    segments = [AudioSegment(timeseries, sr) for timeseries in ts_list]
    declick_ms = 10
    
    overlap_frames = int((declick_ms * sr) / 1000)
    expected_length = sum([len(seg.timeseries) - overlap_frames for seg in segments]) + overlap_frames

    result_segment = Util.concatenate_audio(
        segments,
        declick_fn='linear',
        declick_ms=declick_ms
    )
    actual_length = len(result_segment.timeseries)
    
    assert actual_length == expected_length, \
        f"Concatenated audio has incorrect length! Expected {expected_length}, but got {actual_length}"

def concatenate_audio_with_empty_list():
    """
    Test concatenating an empty list of audio segments
    """
    segments = []
    result_segment = Util.concatenate_audio(
        segments,
        declick_fn='linear',
        declick_ms=10
    )
    assert len(result_segment.timeseries) == 0

def concatenate_audio_with_different_sample_rates():
    """
    Test concatenating audio segments with different sample rates
    """
    sr = 44100
    n_segments = 4
    ts_list = [np.random.rand(sr * 1) for _ in range(n_segments)] # 1 second of audio
    segments = [AudioSegment(timeseries, sr) for timeseries in ts_list]
    segments[0].sample_rate = 22050
    
    with pytest.raises(ValueError):
        Util.concatenate_audio(segments, declick_fn='linear', declick_ms=10)

def test_linear_declick_in():
    """
    Test linear declicking-in
    """
    timeseries = np.ones(10) * 2
    declicked = Util.declick_in(timeseries, 5, 'linear')
    expected_timeseries = np.array([0., 0.5, 1., 1.5, 2., 2. ,2., 2., 2., 2.])

    assert np.array_equal(declicked, expected_timeseries)
    # assert declicking-in does not modify the original timeseries
    assert np.array_equal(timeseries, np.ones(10) * 2)

def test_linear_declick_out():
    """
    Test linear declicking-out
    """
    timeseries = np.ones(10) * 2
    declicked = Util.declick_out(timeseries, 5, 'linear')
    expected_timeseries = np.array([2., 2., 2., 2., 2., 2., 1.5, 1., 0.5, 0.])

    assert np.array_equal(declicked, expected_timeseries)
    # assert declicking-out does not modify the original timeseries
    assert np.array_equal(timeseries, np.ones(10) * 2)

def test_sigmoid_declick_in():
    """
    Test sigmoid declicking-in
    """
    timeseries = np.ones(10) * 2
    declicked = Util.declick_in(timeseries, 5, 'sigmoid')
    expected_timeseries = np.array([0.001, 0.046, 1.0, 1.954, 1.999, 2., 2., 2., 2., 2.])
    
    # assert result is close to expected
    assert np.allclose(declicked, expected_timeseries, atol=0.1)
    # assert declicking-in does not modify the original timeseries
    assert np.array_equal(timeseries, np.ones(10) * 2)

def test_sigmoid_declick_out():
    """
    Test sigmoid declicking-out
    """
    timeseries = np.ones(10) * 2
    declicked = Util.declick_out(timeseries, 5, 'sigmoid')
    expected_timeseries = np.array([2., 2., 2., 2., 2., 1.999, 1.954, 1.0, 0.046, 0.001])
    
    # assert result is close to expected
    assert np.allclose(declicked, expected_timeseries, atol=0.1)
    # assert declicking-out does not modify the original timeseries
    assert np.array_equal(timeseries, np.ones(10) * 2)

def test_declick_in_place():
    """
    Test declicking-in in place
    """
    timeseries = np.ones(10) * 2
    Util.declick_in(timeseries, 5, 'linear', in_place=True)
    expected_timeseries = np.array([0., 0.5, 1., 1.5, 2., 2. ,2., 2., 2., 2.])

    assert np.array_equal(timeseries, expected_timeseries)

