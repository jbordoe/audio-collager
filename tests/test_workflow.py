from unittest.mock import patch, MagicMock
from audio_collage.workflow import create_collage_from_files
from audio_collage.collager import Collager

@patch('audio_collage.workflow.AudioSegment.from_file')
@patch('audio_collage.workflow.Collager.create_collage')
def test_create_collage_from_files(mock_create_collage, mock_from_file):
    """
    Test that the workflow function calls the underlying functions correctly.
    """
    target_file = "target.wav"
    sample_file = "sample.wav"
    outpath = "output.wav"
    declick_fn = Collager.DeclickFn.sigmoid
    declick_ms = 20
    distance_fn = Collager.DistanceFn.mfcc

    mock_audio_segment = MagicMock()
    mock_from_file.return_value = mock_audio_segment
    
    mock_output_audio = MagicMock()
    mock_create_collage.return_value = mock_output_audio

    create_collage_from_files(
        target_file,
        sample_file,
        outpath,
        declick_fn,
        declick_ms,
        distance_fn
    )

    assert mock_from_file.call_count == 2
    mock_from_file.assert_any_call(target_file)
    mock_from_file.assert_any_call(sample_file)
    mock_create_collage.assert_called_once_with(
        target_audio=mock_audio_segment,
        sample_audio=mock_audio_segment,
        declick_fn=declick_fn,
        declick_ms=declick_ms,
        distance_fn=distance_fn
    )
    mock_output_audio.to_file.assert_called_once_with(outpath)
