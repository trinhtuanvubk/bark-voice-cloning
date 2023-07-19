import os
import random

import numpy
from scipy.io import wavfile

from bark.generation import load_model, SAMPLE_RATE
from bark.api import semantic_to_waveform

output = 'output'
output_wav = 'output_wav'

if not os.path.isdir(output):
    raise Exception('No \'output\' folder, make sure you run create_data.py first!')
if not os.path.isdir(output_wav):
    os.mkdir(output_wav)

print('Loading coarse model')
load_model(use_gpu=True, use_small=False, force_reload=False, model_type='coarse')
print('Loading fine model')
load_model(use_gpu=True, use_small=False, force_reload=False, model_type='fine')

for f in os.listdir(output):
    real_name = '.'.join(f.split('.')[:-1])  # Cut off the extension
    file_name = os.path.join(output, f)
    out_file = os.path.join(output_wav, f'{real_name}.wav')
    if not os.path.isfile(out_file) and os.path.isfile(file_name):  # Don't process files that have already been processed
        print(f'Processing {f}')
        wav = semantic_to_waveform(numpy.load(file_name), temp=round(random.uniform(0.6, 0.8), ndigits=2))
        wavfile.write(out_file, SAMPLE_RATE, wav)

print('Done!')