#!/usr/bin/python

import logging
import os
import typer
from typing import Any, List
from rich.logging import RichHandler

from .cli_progress import CLIProgress
from .collager_config import CollagerConfig
from . import workflow

DeclickFn = CollagerConfig.DeclickFn
DistanceFn = CollagerConfig.DistanceFn


app = typer.Typer()

def setup_logging(log_level: str = "INFO"):
    log_level = log_level.upper()
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
    )

def comma_separated_ints(value: Any) -> List[int]:
    return value if isinstance(value, list) else [int(x) for x in value.split(',')]

@app.command()
def collage(
    target_file: str = typer.Option(..., "--target", "-t", help="Path of file to be replicated."),
    sample_file: str = typer.Option(..., "--sample", "-s", help="Path of file to be sampled."),
    outpath: str = typer.Option('./collage.wav', "--outpath", "-o", help="Path of output file."),
    step_ms: int = typer.Option(None, "--step-ms", help="Step size of sample chops in milliseconds"),
    step_factor: float = typer.Option(None, "--step-factor", help="Step size of sample chops as a factor of window size"),
    declick_fn: DeclickFn = typer.Option(..., "--declick-fn", "-f", help="Declicking function."),
    declick_ms: int = typer.Option(0, "--declick-ms", "-d", help="Declick interval in milliseconds."),
    windows: str = typer.Option(
        "500,200,100,50",
        "--windows",
        "-w",
        callback=comma_separated_ints,
        help="List of window sizes (in ms) to use when sampling."
    ),
    distance_fn: DistanceFn = typer.Option(
        DistanceFn.mfcc,
        "--distance-fn",
        "-e",
        help="""Distance function to use when selecting samples.
        Options are:
        - mfcc (default): euclidean distance of mfccs. Slowest but more accurate.
        - fast_mfcc: euclidean distance of mfccs, with padding. Faster but less accurate.
        - mfcc_cosine: cosine distance of mfccs.
        - mean_mfcc: distance of mean mfccs. Fastest but least accurate.
        """
    ),
    log_level: str = typer.Option(
        None,
        "--log-level",
        "--log",
        help="Set logging level. Overrides LOG_LEVEL env var."
    )
) -> None:
    """
    Create a collage based on a given audio file using snippets from another.
    This is a thin wrapper around the create_collage function.
    """
    level = os.getenv("LOG_LEVEL", log_level or "INFO")
    setup_logging(level)

    progress = CLIProgress()

    config = CollagerConfig(
        target_file=target_file,
        sample_file=sample_file,
        outpath=outpath,
        step_ms=step_ms,
        step_factor=step_factor,
        declick_fn=declick_fn,
        declick_ms=declick_ms,
        distance_fn=distance_fn,
        windows=windows,
        progress_callback=progress.update
    )
    workflow.create_collage_from_files(config)

@app.command()
def chop(
    chop_length: int = typer.Option(500, "--length", "-l", help="Length of snippets in milliseconds"),
    step_ms: int = typer.Option(None, "--step-ms", help="Step size of sample chops in milliseconds"),
    step_factor: float = typer.Option(None, "--step-factor", help="Step size of sample chops as a factor of window size"),
    input_filepath: str = typer.Option(..., "--file", "-f", help="Path of file to be chopped."),
    outdir: str = typer.Option(..., "--outdir", "-o", help="Path of directory to write snippets.")
) -> None:
    """
    Chop up a .wav file
    """
    level = os.getenv("LOG_LEVEL", "INFO")
    setup_logging(level)
    progress = CLIProgress()

    workflow.chop_and_write_from_file(
        input_filepath,
        outdir,
        chop_length,
        step_ms=step_ms,
        step_factor=step_factor,
        progress_callback=progress.update
    )

@app.command()
def example() -> None:
    """
    Create an example collage using Amen Brother and Zimba Ku breakbeats.
    """
    level = os.getenv("LOG_LEVEL", "INFO")
    setup_logging(level)
    progress = CLIProgress()

    config = CollagerConfig(
        target_file='./docs/audio/breaks/amen_brother.wav',
        sample_file='./docs/audio/breaks/black_heat__zimba_ku.wav',
        outpath='./collage.wav',
        step_ms=None,
        step_factor=0.5,
        declick_fn=DeclickFn.sigmoid,
        declick_ms=15,
        distance_fn=DistanceFn.fast_mfcc,
        windows=[800, 400, 200, 100],
        progress_callback=progress.update
    )
    workflow.create_collage_from_files(config)

if __name__ == "__main__":
    app()
