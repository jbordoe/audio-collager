#!/usr/bin/python

import sys
import typer
from rich import print
from rich.progress import track
from typing import Callable

import numpy as np

from .util import Util
from .audio_mapper import AudioMapper
from .audio_file import AudioFile
from .collager import Collager

DeclickFn = Collager.DeclickFn
DistanceFn = Collager.DistanceFn

app = typer.Typer()


@app.command()
def collage(
    target_file: str = typer.Option(..., "--target", "-t", help="Path of file to be replicated."),
    sample_file: str = typer.Option(..., "--sample", "-s", help="Path of file to be sampled."),
    outpath: str = typer.Option('./collage.wav', "--outpath", "-o", help="Path of output file."),
    declick_fn: DeclickFn = typer.Option(..., "--declick-fn", "-f", help="Declicking function."),
    declick_ms: int = typer.Option(0, "--declick-ms", "-d", help="Declick interval in milliseconds."),
    distance_fn: DistanceFn = typer.Option(
        DistanceFn.mfcc,
        "--distance-fn",
        "-e",
        help="""Distance function to use when selecting samples.
        Options are:
        - mfcc (default): euclidean distance of mfccs. Slowest but more accurate.
        - fast_mfcc: euclidean distance of mfccs, with padding. Faster but less accurate.
        - mean_mfcc: distance of mean mfccs. Fastest but least accurate.
        """
    )
):
    """
    Create a collage based on a given audio file using snippets from another.
    This is a thin wrapper around the create_collage function.
    """
    Collager.create_collage(
        target_file=target_file,
        sample_file=sample_file,
        outpath=outpath,
        declick_fn=declick_fn,
        declick_ms=declick_ms,
        distance_fn=distance_fn
    )

@app.command()
def chop(
    chop_length: int = typer.Option(500, "--length", "-l", help="Length of snippets in milliseconds"),
    input_filepath: str = typer.Option(..., "--file", "-f", help="Path of file to be chopped."),
    outdir: str = typer.Option(..., "--outdir", "-o", help="Path of directory to write snippets.")
):
    """
    Chop up a .wav file
    """
    input_audio = Util.read_audio(input_filepath)
    slices = Util.chop_audio(input_audio, chop_length)

    for i in track(range(0, len(slices)), description=f'[cyan]Chopping [cyan bold]{input_filepath}[cyan]...'):
        outfile_path = outdir + '/' + str(i).zfill(4) + '.wav'
        audio_slice = slices[i]
        Util.save_audio(audio_slice, outfile_path)

@app.command()
def example():
    """
    Create an example collage using Amen Brother and Zimba Ku breakbeats.
    """
    Collager.create_collage(
        target_file='./docs/audio/breaks/amen_brother.wav',
        sample_file='./docs/audio/breaks/black_heat__zimba_ku.wav',
        outpath='./collage.wav',
        declick_fn=DeclickFn.sigmoid,
        declick_ms=20,
        distance_fn=DistanceFn.fast_mfcc
    )

if __name__ == "__main__":
    app()
