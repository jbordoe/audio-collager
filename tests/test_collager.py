import pytest
from unittest.mock import MagicMock, patch, ANY
from audio_collage.collager import Collager
from audio_collage.audio_segment import AudioSegment

@patch('audio_collage.collager.Util.concatenate_audio')
@patch('audio_collage.collager.AudioMapper')
def test_create_collage_success(mock_audio_mapper, mock_concatenate_audio):
    """
    Test that a collage is created successfully
    """
    target_audio = MagicMock(spec=AudioSegment)
    sample_audio = MagicMock(spec=AudioSegment)
    
    mock_audio_mapper.return_value.map_audio.return_value = [MagicMock(), MagicMock()]
    mock_concatenate_audio.return_value = MagicMock(spec=AudioSegment)

    declick_fn = Collager.DeclickFn.sigmoid
    declick_ms = 20
    distance_fn = Collager.DistanceFn.mfcc

    result = Collager.create_collage(
        target_audio,
        sample_audio,
        declick_fn,
        declick_ms,
        distance_fn
    )

    mock_audio_mapper.assert_called_once_with(sample_audio, target_audio, distance_fn=ANY)
    mock_concatenate_audio.assert_called_once()
    assert isinstance(result, AudioSegment)


@patch('audio_collage.collager.Util.concatenate_audio')
@patch('audio_collage.collager.AudioMapper')
def test_create_collage_no_declick(mock_audio_mapper, mock_concatenate_audio):
    """
    Test that a collage is created successfully with no declicking
    """
    target_audio = MagicMock(spec=AudioSegment)
    sample_audio = MagicMock(spec=AudioSegment)
    
    mock_audio_mapper.return_value.map_audio.return_value = [MagicMock(), MagicMock()]
    mock_concatenate_audio.return_value = MagicMock(spec=AudioSegment)

    declick_fn = None
    declick_ms = 0
    distance_fn = Collager.DistanceFn.mfcc

    Collager.create_collage(
        target_audio,
        sample_audio,
        declick_fn,
        declick_ms,
        distance_fn
    )

    # Check that declick_ms is 0 when declick_fn is None
    call_args, call_kwargs = mock_concatenate_audio.call_args
    assert call_kwargs['declick_ms'] == 0

def test_create_collage_invalid_distance_fn():
    """
    Test that an error is raised when an invalid distance function is provided
    """
    target_audio = MagicMock(spec=AudioSegment)
    sample_audio = MagicMock(spec=AudioSegment)
    declick_fn = Collager.DeclickFn.sigmoid
    declick_ms = 20
    distance_fn = MagicMock()
    distance_fn.value = "invalid"

    with pytest.raises(ValueError):
        Collager.create_collage(
            target_audio,
            sample_audio,
            declick_fn,
            declick_ms,
            distance_fn
        )
