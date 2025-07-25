from typer.testing import CliRunner
from unittest.mock import patch
from audio_collage.cli import app
from audio_collage.collager import CollagerConfig

runner = CliRunner()

@patch('audio_collage.cli.CLIProgress')
@patch('audio_collage.cli.CollagerConfig')
@patch('audio_collage.cli.workflow.create_collage_from_files')
def test_collage_command(
    mock_create_collage_from_files,
    mock_collager_config,
    mock_cli_progress
):
    """
    Test that collage command invokes workflow with the correct arguments.
    """
    target_file = "target.wav"
    sample_file = "sample.wav"
    outpath = "output.wav"
    declick_fn = "sigmoid"
    declick_ms = "20"
    distance_fn = "mfcc"
    step_factor = "0.2"
    windows = "100,200,300"

    result = runner.invoke(app, [
        "collage",
        "--target", target_file,
        "--sample", sample_file,
        "--outpath", outpath,
        "--declick-fn", declick_fn,
        "--declick-ms", declick_ms,
        "--distance-fn", distance_fn,
        "--step-factor", step_factor,
        "--windows", windows
    ])

    assert result.exit_code == 0


    mock_collager_config.assert_called_once_with(
        target_file=target_file,
        sample_file=sample_file,
        outpath=outpath,
        declick_fn=CollagerConfig.DeclickFn[declick_fn],
        declick_ms=int(declick_ms),
        distance_fn=CollagerConfig.DistanceFn[distance_fn],
        step_ms=None,
        step_factor=float(step_factor),
        windows=[100, 200, 300],
        progress_callback=mock_cli_progress.return_value.update
    )

    # Assert a CollagerConfig was passed to workflow
    mock_create_collage_from_files.assert_called_once_with(
        mock_collager_config.return_value
    )

@patch('audio_collage.cli.CLIProgress')
@patch('audio_collage.cli.workflow.chop_and_write_from_file')
def test_chop_command(
    mock_workflow,
    mock_cli_progress
):
    """
    Test that the chop command calls Util.chop_audio with the correct arguments.
    """
    chop_length = 500
    input_filepath = "input.wav"
    outdir = "output_dir"

    result = runner.invoke(app, [
        "chop",
        "--length", str(chop_length),
        "--file", input_filepath,
        "--outdir", outdir,
        "--step-factor", "0.5"
    ])

    assert result.exit_code == 0
    mock_workflow.assert_called_once_with(
        input_filepath,
        outdir,
        chop_length,
        step_ms=None,
        step_factor=0.5,
        progress_callback=mock_cli_progress.return_value.update,
    )

@patch('audio_collage.cli.CLIProgress')
@patch('audio_collage.cli.CollagerConfig')
@patch('audio_collage.cli.workflow.create_collage_from_files')
def test_example_command(
    mock_create_collage_from_files,
    mock_config_init,
    mock_cli_progress
):
    """
    Test that the example command invokes workflow with the correct arguments.
    """
    result = runner.invoke(app, ["example"])

    assert result.exit_code == 0

    mock_config_init.assert_called_once_with(
        target_file='./docs/audio/breaks/amen_brother.wav',
        sample_file='./docs/audio/breaks/black_heat__zimba_ku.wav',
        outpath='./collage.wav',
        declick_fn=CollagerConfig.DeclickFn.sigmoid,
        declick_ms=15,
        distance_fn=CollagerConfig.DistanceFn.fast_mfcc,
        step_ms=None,
        step_factor=0.5,
        windows=[800, 400, 200, 100],
        progress_callback=mock_cli_progress.return_value.update
    )

    mock_create_collage_from_files.assert_called_once_with(
        mock_config_init.return_value
    )
