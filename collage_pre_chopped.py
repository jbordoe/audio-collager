#!/usr/bin/python

import sys
import typer

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

def main(
    target_dir: str = typer.Option(..., "--target-dir", "-t", help="Path of directory containing target snippets."),
    sample_dir: str = typer.Option(..., "--sample-dir", "-s", help="Path of directory containing sample snippets."),
    outpath: str = typer.Option('./collage.wav', "--outpath", "-o", help="Path of output file.")
):
    """
    Create a collage based on a pre-chopped audio file using snippets
    """
    tmp_dir = './tmp'

    sample_files = list_files(sample_dir)
    target_files = list(list_files(target_dir))
    target_files.sort()

    sample_files = [extract_features(path) for path in sample_files]

    selected_snippets = []

    for target_path in target_files:
        y1, sr1 = librosa.load(target_path)
        mfcc = librosa.feature.mfcc(y1,sr1) #Computing MFCC values

        min_dist = 999999999
        closest_sample = None
        for sample_path, sample_mfcc in sample_files:
            dist, cost, acc_cost, path = dtw(mfcc.T, sample_mfcc.T, dist=lambda x, y: norm(x - y, ord=1))
            if dist < min_dist:
                min_dist = dist
                closest_sample = sample_path

        selected_snippets.append(closest_sample)

    with wave.open(outpath, 'wb') as wav_out:
        for wav_path in selected_snippets:
            with wave.open(wav_path, 'rb') as wav_in:
                if not wav_out.getnframes():
                    wav_out.setparams(wav_in.getparams())
                wav_out.writeframes(wav_in.readframes(wav_in.getnframes()))

if __name__ == "__main__":
    typer.run(main)
