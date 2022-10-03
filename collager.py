#!/usr/bin/python

import sys
import typer
from strenum import StrEnum
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn, track

import vptree
import numpy as np

from lib import util

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

    sample_audio = util.Util.read_audio(sample_file)

    samples = {}

    windows = [500,200,100,50]
    windows = [i + declick_ms for i in windows]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=False,
    ) as progress:
        progress.add_task(description="[cyan]Chopping and analysing sample audio...", total=None)
        for window in windows:
            sample_group = util.Util.chop_audio(sample_audio, window)

            for s in sample_group:
                util.Util.extract_features(s)

            tree = vptree.VPTree(sample_group, util.Util.mfcc_dist)
            samples[window] = tree

    selected_snippets = []

    target_audio = util.Util.read_audio(target_file)
    util.Util.extract_features(target_audio)
    target_sr = target_audio.sample_rate

    pointer = 0
    with Progress() as progress:
        task = progress.add_task('[cyan]Selecting samples...', total=100)
        while pointer < target_audio.timeseries.size:
            pct_complete = (pointer / target_audio.timeseries.size) * 100 if pointer else 0
            progress.update(task, completed=pct_complete)


            best_snippet = None
            best_snippet_dist = 999999
            best_snippet_window = None
            for window in windows:
                window_size_frames = int((window / 1000) * target_sr)
                target_chunk = util.Util.AudioFile(
                        target_audio.timeseries[pointer:pointer + window_size_frames - 1],
                        target_sr,
                        )
                util.Util.extract_features(target_chunk)

                group = samples[window]
                nearest_dist, nearest = group.get_nearest_neighbor(target_chunk)

                if nearest_dist < best_snippet_dist:
                    best_snippet_dist = nearest_dist
                    best_snippet = nearest
                    best_snippet_window = window_size_frames

            selected_snippets.append(best_snippet)
            pointer += best_snippet_window - int((declick_ms /1000) * target_sr)
        progress.update(task, completed=100)

    output_data = []
    i = 0
    for snippet in track(selected_snippets, description="[cyan]Concatenating samples..."):
        if declick_fn:
            snippet = util.Util.declick(snippet, declick_fn, declick_ms)

        x = snippet.timeseries
        if declick_ms and output_data and i < len(selected_snippets)-1:
            overlap_frames = int((declick_ms * snippet.sample_rate) / 1000)
            overlap = np.add(output_data[-overlap_frames:], x[:overlap_frames])
            output_data = output_data[:-overlap_frames]
            x = np.concatenate([overlap, x[overlap_frames:]])

        output_data.extend(x)
        i += 1

    print(f'[cyan]Saving collage to [yellow]{outpath}[cyan]...')
    output_audio = util.Util.AudioFile(output_data, sample_audio.sample_rate)
    util.Util.save_audio(output_audio, outpath)

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
    input_audio = util.Util.read_audio(input_filepath)
    slices = util.Util.chop_audio(input_audio, chop_length)

#    conn = sqlite3.connect('db/audio.db')
#    cursor = conn.cursor()

    for i in track(range(0, len(slices)), description=f'[cyan]Chopping [cyan bold]{input_filepath}[cyan]...'):
        outfile_path = outdir + '/' + str(i).zfill(4) + '.wav'

#        mfcc_json = None
        audio_slice = slices[i]
        util.Util.save_audio(audio_slice, outfile_path)
#        if analyse:
#            data = ausio_slice.timeseries
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

if __name__ == "__main__":
    app()
