import torchaudio
import torch
from transformers import BertTokenizer
from scipy.io.wavfile import write as write_wav

from encodec.utils import convert_audio
from bark.api import generate_audio
from bark.generation import (
    SAMPLE_RATE,
    preload_models,
    codec_decode,
    generate_coarse,
    generate_fine,
    generate_text_semantic,
    load_codec_model)

def clone_voice(args):

    model = load_codec_model(use_gpu=True if args.device=='cuda' else False)

    # Enter your prompt and speaker here
   
    preload_models(
            text_use_gpu=True,
            text_use_small=False,
            coarse_use_gpu=True,
            coarse_use_small=False,
            fine_use_gpu=True,
            fine_use_small=False,
            codec_use_gpu=True,
            force_reload=False,
            path=args.saved_models
        )
    
    audio_array = generate_audio(
            args.text_prompt,
            history_prompt=args.speaker_name,
            text_temp=args.text_temp,
            waveform_temp=args.waveform_temp)
    

    write_wav(args.cloned_output, SAMPLE_RATE, audio_array)