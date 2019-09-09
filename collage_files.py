#!/usr/bin/python

import sys
import argparse

import vptree

from lib import util

parser=argparse.ArgumentParser(description='Create a collage based on a given audio file based on snippets from another')
parser.add_argument(
    '-i', '--source_file', type=str, required=True, help='Path of file to be replicated')
parser.add_argument(
    '-s', '--sample_file', type=str, required=True, help='Path of file to be sampled.')
parser.add_argument(
    '-o', '--outpath', type=str, required=False, default='./collage.wav', help='Path of output file.')

args = parser.parse_args()

sourcefile = args.source_file
samplefile = args.sample_file
outpath    = args.outpath

sample_audio = util.Util.read_audio(samplefile)

samples = {}

windows = [500,200,100,50]

print('Chopping sample file.')
for window in windows:
    sample_group = util.Util.chop_audio(sample_audio, window)

    for s in sample_group:
        util.Util.extract_features(s)

    tree = vptree.VPTree(sample_group, util.Util.audio_dist)
    samples[window] = tree

selected_snippets = []

source_audio = util.Util.read_audio(sourcefile)
util.Util.extract_features(source_audio)
source_sr = source_audio.sample_rate

pointer = 0
print('Generating collage with samples.')
while pointer < source_audio.timeseries.size:
    pct_complete = int((pointer / source_audio.timeseries.size) * 100) if pointer else 0
    pct_remaining = 100 - pct_complete

    sys.stdout.write('\r')
    sys.stdout.write('▓'*pct_complete + '_'*pct_remaining + '{}%'.format(pct_complete))
    sys.stdout.flush()

    best_snippet = None
    best_snippet_dist = 999999
    best_snippet_window = None
    for window in windows:
        window_size_frames = int((window / 1000) * source_sr)
        source_chunk = util.Util.AudioFile(
                source_audio.timeseries[pointer:pointer + window_size_frames - 1],
                source_sr,
                )
        util.Util.extract_features(source_chunk)

        group = samples[window]
        nearest_dist, nearest = group.get_nearest_neighbor(source_chunk)

        if nearest_dist < best_snippet_dist:
            best_snippet_dist = nearest_dist
            best_snippet = nearest
            best_snippet_window = window_size_frames

    selected_snippets.append(best_snippet)
    pointer += best_snippet_window


sys.stdout.write('\r')
print('Collage generated.')

output_data = []
for snippet in selected_snippets:
    output_data.extend(snippet.timeseries)

print('Saving collage file.')
output_audio = util.Util.AudioFile(output_data, sample_audio.sample_rate)
util.Util.save_audio(output_audio, outpath)

print('Done!')
