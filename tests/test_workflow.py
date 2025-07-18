from unittest.mock import patch, MagicMock
from audio_collage.workflow import create_collage_from_files, chop_and_write_from_file
from audio_collage.collager_config import CollagerConfig

@patch('audio_collage.cli_progress.CLIProgress')
@patch('audio_collage.workflow.AudioSegment.from_file')
@patch('audio_collage.workflow.Collager.create_collage')
def test_create_collage_from_files(
    mock_create_collage,
    mock_from_file,
    mock_progress
):
    """
    Test that the workflow function calls the underlying functions correctly.
    """
    target_file = "target.wav"
    sample_file = "sample.wav"
    outpath = "output.wav"
    declick_fn = CollagerConfig.DeclickFn.sigmoid
    declick_ms = 20
    distance_fn = CollagerConfig.DistanceFn.mfcc
    step_factor = 0.2

    config = CollagerConfig(
        target_file=target_file,
        sample_file=sample_file,
        outpath=outpath,
        declick_fn=declick_fn,
        declick_ms=declick_ms,
        distance_fn=distance_fn,
        step_factor=step_factor
    )
    mock_audio_segment = MagicMock()
    mock_from_file.return_value = mock_audio_segment

    mock_output_audio = MagicMock()
    mock_create_collage.return_value = mock_output_audio

    create_collage_from_files(config)

    assert mock_from_file.call_count == 2
    mock_from_file.assert_any_call(target_file)
    mock_from_file.assert_any_call(sample_file)
    mock_create_collage.assert_called_once_with(
        target_audio=mock_audio_segment,
        sample_audio=mock_audio_segment,
        config=config
    )
    mock_output_audio.to_file.assert_called_once_with(outpath)

@patch('audio_collage.audio_segment.AudioSegment.from_file')
@patch('audio_collage.util.Util.chop_audio')
def test_chop_and_write_from_file(
    mock_chop_audio,
    mock_from_file,
):
    """
    Test that functions calls Util.chop_audio with the correct arguments.
    """
    chop_length = 500
    input_filepath = "input.wav"
    outdir = "output_dir"

    mock_callback = MagicMock()
    mock_slices = [MagicMock(), MagicMock()]
    mock_chop_audio.return_value = mock_slices

    chop_and_write_from_file(
        input_filepath,
        outdir,
        chop_length,
        step_ms=None,
        step_factor=0.5,
        progress_callback=mock_callback
    )

    mock_from_file.assert_called_once_with(input_filepath)
    mock_chop_audio.assert_called_once_with(
        mock_from_file.return_value,
        chop_length,
        step_ms=None,
        step_factor=0.5,
        progress_callback=mock_callback
    )
    for i, mock_slice in enumerate(mock_slices):
        mock_slice.to_file.assert_called_once_with(f"{outdir}/{chop_length}ms.{i:04}.wav")

