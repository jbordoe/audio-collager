#!/usr/bin/python

import sys
import argparse

from os import listdir
from os.path import isfile, join
import re

from scipy.io import wavfile as wav
from scipy.fftpack import fft
import numpy as np

def list_files(directory):
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f)) and re.match(r'.*\.wav$',f)]
    return map(lambda f: join(directory, f), onlyfiles)

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
snippet_files = list_files(chop_dir)

r, d = wav.read(sample_files[0])
print fft(d).size

for sample in sample_files:
    pass
