#!/usr/bin/python

import sys
import typer
from strenum import StrEnum
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn, track

import numpy as np

from .util import Util
from .collager import Collager

DeclickFn = StrEnum('Declickfn', {k: k for k in ['sigmoid', 'linear']})

app = typer.Typer()

@app.command()
def collage(
    target_file: str = typer.Option(..., "--target", "-t", help="Path of file to be replicated."),
    sample_file: str = typer.Option(..., "--sample", "-s", help="Path of file to be sampled."),
    outpath: str = typer.Option('./collage.wav', "--outpath", "-o", help="Path of output file."),
    declick_fn: DeclickFn = typer.Option(..., "--declick-fn", "-f", help="Declicking function."),
    declick_ms: int = typer.Option(0, "--declick-ms", "-d", help="Declick interval in milliseconds.")
):
    """
    Create a collage based on a given audio file using snippets from another
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

    windows = [500,200,100,50]
    windows = [i + declick_ms for i in windows]
    
    collager = Collager(sample_audio, target_audio)
    selected_snippets = collager.collage(windows=windows, overlap_ms=declick_ms)

    output_audio = Util.concatenate_audio(
        track(selected_snippets, description="[cyan]Concatenating samples..."),
        declick_fn=declick_fn,
        declick_ms=declick_ms
    )

    print(f'[cyan]Saving collage to [yellow]{outpath}[cyan]...')
    Util.save_audio(output_audio, outpath)

    print('[green bold]Done!')

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

#    conn = sqlite3.connect('db/audio.db')
#    cursor = conn.cursor()

    for i in track(range(0, len(slices)), description=f'[cyan]Chopping [cyan bold]{input_filepath}[cyan]...'):
        # TODO; make zfill dynamic
        outfile_path = outdir + '/' + str(i).zfill(4) + '.wav'

#        mfcc_json = None
        audio_slice = slices[i]
        Util.save_audio(audio_slice, outfile_path)
#        if analyse:
#            data = audio_slice.timeseries
#            sample_rate = audio_slice.sample_rate
#            mfcc = librosa.feature.mfcc(data, sample_rate) #Computing MFCC values
#            mfcc_json = json.dumps(mfcc.tolist())
#
#        conn.execute(
#                'INSERT INTO samples (source, path, features) VALUES (?, ?, ?)',
#                ['misc', outfile_path, mfcc_json]
#                )

#    conn.commit()
#    conn.close()

@app.command()
def example():
    """
    Create an example collage using Amen Brother and Zimba Ku breakbeats.
    """
    collage(
        target_file='./docs/audio/breaks/amen_brother.wav',
        sample_file='./docs/audio/breaks/black_heat__zimba_ku.wav',
        outpath='./collage.wav',
        declick_fn='sigmoid',
        declick_ms=20
    )

if __name__ == "__main__":
    app()
