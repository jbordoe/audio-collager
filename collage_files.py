#!/usr/bin/python

import sys
import typer
from strenum import StrEnum

import vptree
import numpy as np

from lib import util

DeclickFn = StrEnum('Declickfn', {k: k for k in ['sigmoid', 'linear']})

def main(
    target_file: str = typer.Option(..., "--target", "-t", help="Path of file to be replicated."),
    sample_file: str = typer.Option(..., "--sample", "-s", help="Path of file to be sampled."),
    outpath: str = typer.Option('./collage.wav', "--outpath", "-o", help="Path of output file."),
    decklick_fn: DeclickFn = typer.Option(..., "--declick-fn", "-f", help="Declicking function."),
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
        declick_ms = args.declick_ms or default_dc_ms[declick_fn]
    else:
        declick_ms = 0

    sample_audio = util.Util.read_audio(sample_file)

    samples = {}

    windows = [500,200,100,50]
    windows = [i + declick_ms for i in windows]

    print('Chopping sample file.')
    for window in windows:
        sample_group = util.Util.chop_audio(sample_audio, window)

        for s in sample_group:
            util.Util.extract_features(s)

        tree = vptree.VPTree(sample_group, util.Util.audio_dist)
        samples[window] = tree

    selected_snippets = []

    target_audio = util.Util.read_audio(target_file)
    util.Util.extract_features(target_audio)
    target_sr = target_audio.sample_rate

    pointer = 0
    print('Generating collage with samples.')
    while pointer < target_audio.timeseries.size:
        pct_complete = int((pointer / target_audio.timeseries.size) * 100) if pointer else 0
        pct_remaining = 100 - pct_complete

        sys.stdout.write('\r')
        sys.stdout.write('▓'*pct_complete + '_'*pct_remaining + '{}%'.format(pct_complete))
        sys.stdout.flush()

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


    sys.stdout.write('\r')
    print('Collage generated.')

    output_data = []
    i = 0
    for snippet in selected_snippets:
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

    print('Saving collage file.')
    output_audio = util.Util.AudioFile(output_data, sample_audio.sample_rate)
    util.Util.save_audio(output_audio, outpath)

    print('Done!')

if __name__ == "__main__":
    typer.run(main)
