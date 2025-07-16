from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from audio_collage.cli import app

runner = CliRunner()

@patch('audio_collage.cli.Collager.create_collage')
def test_collage_command(mock_create_collage):
    """
    Test that the collage command calls Collager.create_collage with the correct arguments.
    """
    target_file = "target.wav"
    sample_file = "sample.wav"
    outpath = "output.wav"
    declick_fn = "sigmoid"
    declick_ms = "20"
    distance_fn = "mfcc"

    result = runner.invoke(app, [
        "collage",
        "--target", target_file,
        "--sample", sample_file,
        "--outpath", outpath,
        "--declick-fn", declick_fn,
        "--declick-ms", declick_ms,
        "--distance-fn", distance_fn,
    ])

    assert result.exit_code == 0
    mock_create_collage.assert_called_once()
    
    from audio_collage.collager import Collager
    expected_declick_fn = Collager.DeclickFn[declick_fn]
    expected_distance_fn = Collager.DistanceFn[distance_fn]

    mock_create_collage.return_value.to_file.assert_called_once_with(outpath)
    mock_create_collage.assert_called_once_with(
        target_file=target_file,
        sample_file=sample_file,
        declick_fn=expected_declick_fn,
        declick_ms=int(declick_ms),
        distance_fn=expected_distance_fn
    )

@patch('audio_collage.cli.AudioSegment.from_file')
@patch('audio_collage.cli.Util.chop_audio')
def test_chop_command(mock_chop_audio, mock_from_file):
    """
    Test that the chop command calls Util.chop_audio with the correct arguments.
    """
    chop_length = 500
    input_filepath = "input.wav"
    outdir = "output_dir"

    mock_slices = [MagicMock(), MagicMock()]
    mock_chop_audio.return_value = mock_slices 

    result = runner.invoke(app, [
        "chop",
        "--length", str(chop_length),
        "--file", input_filepath,
        "--outdir", outdir,
    ])

    assert result.exit_code == 0
    mock_from_file.assert_called_once_with(input_filepath)
    mock_chop_audio.assert_called_once_with(mock_from_file.return_value, chop_length)
    for i, slice in enumerate(mock_slices):
        slice.to_file.assert_called_once_with(f"{outdir}/{i:04}.wav")


@patch('audio_collage.cli.Collager.create_collage')
def test_example_command(mock_create_collage):
    """
    Test that example command calls Collager.create_collage with correct arguments.
    """
    result = runner.invoke(app, ["example"])

    assert result.exit_code == 0
    mock_create_collage.assert_called_once()
    mock_create_collage.return_value.to_file.assert_called_once_with("./collage.wav")
