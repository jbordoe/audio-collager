#!/usr/bin/python

import sys
import wave

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

DEFAULT_CHOP_LENGTH = 500

input_filepath = sys.argv[1]

chop_length = DEFAULT_CHOP_LENGTH
try:
    chop_length = int(sys.argv[2])
except IndexError:
    pass

infile = wave.open(input_filepath)
infile_data = {
    'infile': infile,
    'rate': infile.getframerate(),
    'frames': infile.getnframes(),
    'width': infile.getsampwidth()
}
infile_data['fpms'] = infile_data['rate'] / 1000
infile_data['duration'] = infile_data['frames'] / float(infile_data['fpms'])

outdir = './samples/chopped'

pointer_pos = 0
files_generated = 0

while pointer_pos < infile_data['duration']:
    outfile_path = outdir + '/' + str(files_generated+1) + '.wav'

    start_ms = pointer_pos
    end_ms = pointer_pos + chop_length
    slice(infile_data, outfile_path, start_ms, end_ms)

    pointer_pos += chop_length
    files_generated += 1

infile.close
