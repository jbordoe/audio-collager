#!/usr/bin/python

import sys
import argparse
import sqlite3
import json
import matplotlib.pyplot as plt

from lib import util

parser = argparse.ArgumentParser(description='Chop up a .wav file.')
parser.add_argument(
    '-l', '--length', type=int, required=False, default=500,
     help='Length of snippets in milliseconds.')
parser.add_argument(
    '-i', '--infile', type=argparse.FileType('r'), required=True, help='Path of source file to be chopped.')
parser.add_argument(
    '-o', '--outdir', type=str, required=False, default='./samples/misc/chopped', help='Path of source file to be chopped.')
parser.add_argument(
    '-a', '--analyse', action='store_true', help='Extract features of each snippet.')

args = parser.parse_args()

chop_length = args.length
input_filepath = args.infile.name
outdir = args.outdir
analyse = args.analyse

input_audio = util.Util.read_audio(input_filepath)
slices = util.Util.chop_audio(input_audio, chop_length)

conn = sqlite3.connect('db/audio.db')
cursor = conn.cursor()

for i in range(0, len(slices)):
    outfile_path = outdir + '/' + str(i).zfill(4) + '.wav'

    mfcc_json = None
    audio_slice = slices[i]
    util.Util.save_audio(audio_slice, outfile_path)
    if analyse:
        data = ausio_slice.timeseries
        sample_rate = audio_slice.sample_rate
        mfcc = librosa.feature.mfcc(data, sample_rate) #Computing MFCC values
        mfcc_json = json.dumps(mfcc.tolist())

    conn.execute(
            'INSERT INTO samples (source, path, features) VALUES (?, ?, ?)',
            ['misc', outfile_path, mfcc_json]
            )

conn.commit()
conn.close()
