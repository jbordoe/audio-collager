#!/usr/bin/python

import typer
from rich.progress import track


from .util import Util
from .audio_segment import AudioSegment
from .collager import Collager
from . import workflow


DeclickFn = Collager.DeclickFn
DistanceFn = Collager.DistanceFn

app = typer.Typer()


@app.command()
def collage(
    target_file: str = typer.Option(..., "--target", "-t", help="Path of file to be replicated."),
    sample_file: str = typer.Option(..., "--sample", "-s", help="Path of file to be sampled."),
    outpath: str = typer.Option('./collage.wav', "--outpath", "-o", help="Path of output file."),
    step_ms: int = typer.Option(None, "--step-ms", help="Step size of sample chops in milliseconds"),
    step_factor: float = typer.Option(None, "--step-factor", help="Step size of sample chops as a factor of window size"),
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
        - mfcc_cosine: cosine distance of mfccs.
        - mean_mfcc: distance of mean mfccs. Fastest but least accurate.
        """
    )
):
    """
    Create a collage based on a given audio file using snippets from another.
    This is a thin wrapper around the create_collage function.
    """
    workflow.create_collage_from_files(
        target_file=target_file,
        sample_file=sample_file,
        outpath=outpath,
        declick_fn=declick_fn,
        declick_ms=declick_ms,
        distance_fn=distance_fn,
        step_ms=step_ms,
        step_factor=step_factor
    )

@app.command()
def chop(
    chop_length: int = typer.Option(500, "--length", "-l", help="Length of snippets in milliseconds"),
    step_ms: int = typer.Option(None, "--step-ms", help="Step size of sample chops in milliseconds"),
    step_factor: float = typer.Option(None, "--step-factor", help="Step size of sample chops as a factor of window size"),
    input_filepath: str = typer.Option(..., "--file", "-f", help="Path of file to be chopped."),
    outdir: str = typer.Option(..., "--outdir", "-o", help="Path of directory to write snippets.")
):
    """
    Chop up a .wav file
    """
    if step_ms and step_factor:
        print("[red]Cannot specify both --step-ms and --step-factor")
        typer.Abort()

    # TODO: move this to the workflow module
    input_audio = AudioSegment.from_file(input_filepath)
    slices = Util.chop_audio(
        input_audio,
        chop_length,
        step_ms=step_ms,
        step_factor=step_factor
    )

    # TODO: keep track call here before passing to workflow
    for i in track(range(0, len(slices)), description=f'[cyan]Chopping [cyan bold]{input_filepath}[cyan]...'):
        outfile_path = outdir + '/' + str(i).zfill(4) + '.wav'
        audio_slice = slices[i]
        audio_slice.to_file(outfile_path)

@app.command()
def example():
    """
    Create an example collage using Amen Brother and Zimba Ku breakbeats.
    """
    workflow.create_collage_from_files(
        sample_file='./docs/audio/breaks/amen_brother.wav',
        target_file='./docs/audio/breaks/black_heat__zimba_ku.wav',
        outpath='./collage.wav',
        declick_fn=DeclickFn.sigmoid,
        declick_ms=15,
        distance_fn=DistanceFn.fast_mfcc,
        step_ms=None,
        step_factor=None
    )

if __name__ == "__main__":
    app()
