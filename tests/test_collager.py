import pytest
from unittest.mock import MagicMock, Mock, patch
from audio_collage.collager import Collager
from audio_collage.audio_segment import AudioSegment

import typer
import numpy as np

@patch('audio_collage.collager.AudioSegment')
@patch('audio_collage.collager.AudioMapper')
def test_create_collage_success(mock_audio_mapper, mock_audio_segment):
    """
    Test that a collage is created successfully
    """
    mock_audio_segment.from_file.return_value = MagicMock()
    mock_audio_mapper.return_value.map_audio.return_value = [
        AudioSegment(timeseries=np.array([1, 2, 3]), sample_rate=44100),
        AudioSegment(timeseries=np.array([4, 5, 6]), sample_rate=44100),
    ]

    target_file = "dummy_target.wav"
    sample_file = "dummy_sample.wav"
    declick_fn = Collager.DeclickFn.sigmoid
    declick_ms = 20
    distance_fn = Collager.DistanceFn.mfcc

    Collager.create_collage(
        target_file,
        sample_file,
        declick_fn,
        declick_ms,
        distance_fn
    )

    mock_audio_segment.from_file.assert_any_call(target_file)
    mock_audio_segment.from_file.assert_any_call(sample_file)
    mock_audio_mapper.assert_called_once()

@patch('audio_collage.collager.AudioSegment')
@patch('audio_collage.collager.AudioMapper')
@patch('audio_collage.collager.Util')
def test_create_collage_no_declick(mock_util, mock_audio_mapper, mock_audio_segment):
    """
    Test that a collage is created successfully with no declicking
    """
    mock_audio_segment.from_file.return_value = MagicMock()
    mock_audio_mapper.return_value.map_audio.return_value = [MagicMock(), MagicMock()]
    mock_util.concatenate_audio.return_value = MagicMock()

    target_file = "dummy_target.wav"
    sample_file = "dummy_sample.wav"
    declick_fn = None
    declick_ms = 0
    distance_fn = Collager.DistanceFn.mfcc

    Collager.create_collage(
        target_file,
        sample_file,
        declick_fn,
        declick_ms,
        distance_fn
    )

    mock_audio_segment.from_file.assert_any_call(target_file)
    mock_audio_segment.from_file.assert_any_call(sample_file)
    mock_audio_mapper.assert_called_once()
    mock_util.concatenate_audio.assert_called_once()

    call_args, call_kwargs = mock_util.concatenate_audio.call_args
    assert call_kwargs['declick_ms'] == 0

def test_create_collage_invalid_distance_fn():
    """
    Test that an error is raised when an invalid distance function is provided
    """
    target_file = "dummy_target.wav"
    sample_file = "dummy_sample.wav"
    declick_fn = Collager.DeclickFn.sigmoid
    declick_ms = 20
    distance_fn = Mock(value='invalid')

    with pytest.raises(ValueError):
        Collager.create_collage(
            target_file,
            sample_file,
            declick_fn,
            declick_ms,
            distance_fn
        )
