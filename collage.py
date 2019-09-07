#!/usr/bin/python

import sys
import argparse

from os import listdir
from os.path import isfile, join
import re

import wave
import librosa
from dtw import dtw
from numpy.linalg import norm

def list_files(directory):
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f)) and re.match(r'.*\.wav$',f)]
    return map(lambda f: join(directory, f), onlyfiles)

def extract_features(path):
    y, sr = librosa.load(path)
    mfcc = librosa.feature.mfcc(y,sr) #Computing MFCC values
    return [path, mfcc]

parser=argparse.ArgumentParser(description='Create a collage based on a given file using given snippets.')
parser.add_argument(
    '-s', '--sampledir', type=str, required=True, help='Path of chopped source sounds directory.')
parser.add_argument(
    '-c', '--chopdir', type=str, required=True, help='Path of snippets directory.')
parser.add_argument(
    '-o', '--outfile', type=str, required=False, default='./collage.wav', help='Path of output file.')

args = parser.parse_args()

sample_dir = args.sampledir
chop_dir   = args.chopdir
outfile    = args.outfile

tmp_dir = './tmp'

sample_files = list_files(sample_dir)
snippet_files = list(list_files(chop_dir))
snippet_files.sort()

sample_files = [extract_features(path) for path in sample_files]

selected_snippets = []

for snippet_path in snippet_files:
    y1, sr1 = librosa.load(snippet_path)
    mfcc = librosa.feature.mfcc(y1,sr1) #Computing MFCC values

    min_dist = 999999999
    closest_sample = None
    for sample_path, sample_mfcc in sample_files:
        dist, cost, acc_cost, path = dtw(mfcc.T, sample_mfcc.T, dist=lambda x, y: norm(x - y, ord=1))
        if dist < min_dist:
            min_dist = dist
            closest_sample = sample_path

    selected_snippets.append(closest_sample)

with wave.open(outfile, 'wb') as wav_out:
    for wav_path in selected_snippets:
        with wave.open(wav_path, 'rb') as wav_in:
            if not wav_out.getnframes():
                wav_out.setparams(wav_in.getparams())
            wav_out.writeframes(wav_in.readframes(wav_in.getnframes()))
