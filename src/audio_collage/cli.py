#!/usr/bin/python

import sys
import typer
from strenum import StrEnum
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn, track
from typing import Callable

import numpy as np

from .util import Util
from .collager import Collager
from .audio_file import AudioFile

DeclickFn = StrEnum('Declickfn', {k: k for k in ['sigmoid', 'linear']})
DistanceFn = StrEnum('DistanceFn', {k: k for k in ['mfcc', 'fast_mfcc', 'mean_mfcc']})

app = typer.Typer()

def create_collage(
    target_file: str,
    sample_file: str,
    outpath: str,
    declick_fn: DeclickFn,
    declick_ms: int,
    distance_fn: DistanceFn
):
    """
    This is the core logic for creating a collage.
    """
    default_dc_ms = {
        'sigmoid': 20,
        'linear': 70,
    }
    if declick_fn:
        declick_ms = declick_ms or default_dc_ms[declick_fn]
    else:
        declick_ms = 0

    sample_audio = Util.read_audio(sample_file)
    target_audio = Util.read_audio(target_file)

    windows = [500, 200, 100, 50]
    windows = [i + declick_ms for i in windows]

    dist_fn_map: Dict[str, Callable[[AudioFile, AudioFile], float]] = {
        'mfcc': Util.mfcc_dist,
        'fast_mfcc': Util.fast_mfcc_dist,
        'mean_mfcc': Util.mean_mfcc_dist,
    }
    
    selected_distance_fn = dist_fn_map.get(distance_fn.value)
    if not selected_distance_fn:
        print(f'[yellow]Invalid distance function [yellow bold]{distance_fn}[yellow]!')
        raise typer.Exit(code=1)

    collager = Collager(sample_audio, target_audio, distance_fn=selected_distance_fn)
    selected_snippets = collager.collage(
        windows=windows,
        overlap_ms=declick_ms,
    )

    output_audio = Util.concatenate_audio(
        track(selected_snippets, description="[cyan]Concatenating samples..."),
        declick_fn=declick_fn,
        declick_ms=declick_ms
    )

    print(f'[cyan]Saving collage to [yellow]{outpath}[cyan]...')
    Util.save_audio(output_audio, outpath)

    print('[green bold]Done!')


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
    create_collage(
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
    create_collage(
        target_file='./docs/audio/breaks/amen_brother.wav',
        sample_file='./docs/audio/breaks/black_heat__zimba_ku.wav',
        outpath='./collage.wav',
        declick_fn=DeclickFn.sigmoid,
        declick_ms=20,
        distance_fn=DistanceFn.fast_mfcc
    )

if __name__ == "__main__":
    app()
