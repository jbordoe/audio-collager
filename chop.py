#!/usr/bin/python

import sys
import wave
import argparse
import sqlite3
import json
import librosa
import matplotlib.pyplot as plt

def slice(infile_data, outfilename, start_ms, end_ms):
    width = infile_data['width']
    rate = infile_data['rate']
    fpms = infile_data['fpms']
    infile = infile_data['infile']

    length = (end_ms - start_ms) * fpms
    start_index = start_ms * fpms

    out = wave.open(outfilename, "w")
    out.setparams((infile.getnchannels(), width, rate, length, infile.getcomptype(), infile.getcompname()))
    
    infile.rewind()
    anchor = infile.tell()
    infile.setpos(anchor + start_index)
    out.writeframes(infile.readframes(length))

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

infile = wave.open(input_filepath, 'rb')
infile_data = {
    'infile': infile,
    'rate': infile.getframerate(),
    'frames': infile.getnframes(),
    'width': infile.getsampwidth()
}
infile_data['fpms'] = infile_data['rate'] // 1000
infile_data['duration'] = infile_data['frames'] // float(infile_data['fpms'])

conn = sqlite3.connect('db/audio.db')
cursor = conn.cursor()

pointer_pos = 0
files_generated = 0

while pointer_pos < infile_data['duration']:
    outfile_path = outdir + '/' + str(files_generated+1) + '.wav'

    start_ms = pointer_pos
    end_ms = pointer_pos + chop_length

    slice(infile_data, outfile_path, start_ms, end_ms)

    pointer_pos += chop_length
    files_generated += 1

    mfcc1_json = None
    if analyse:
        y1, sr1 = librosa.load(outfile_path)
        mfcc1 = librosa.feature.mfcc(y1,sr1) #Computing MFCC values
        mfcc1_json = json.dumps(mfcc1.tolist())

    conn.execute(
            'INSERT INTO samples (source, path, features) VALUES (?, ?, ?)',
            ['misc', outfile_path, mfcc1_json]
            )
infile.close

conn.commit()
conn.close()
