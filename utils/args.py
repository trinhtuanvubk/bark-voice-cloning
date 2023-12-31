import argparse
import torch
from pathlib import Path

def get_args():
    # create parser args
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_prompts', type=str, default='./bark/assets/prompts')
    # senario
    parser.add_argument('--scenario', type=str, default='train')
    
    # generate npz for speaker
    parser.add_argument('--large_model', type=str, default='quantifier_V1_hubert_base_ls960_23.pth')
    parser.add_argument('--large_tokenizer', type=str, default="tokenizer_large.pth")
    parser.add_argument('--base_model', type=str, default='quantifier_hubert_base_ls960_14.pth')
    parser.add_argument('--base_tokenizer', type=str, default="tokenizer.pth")
    parser.add_argument('--use_large_quant_model', action="store_true")
    parser.add_argument('--wav_file', type=str, default='./sample/vietnamtest.wav')
    parser.add_argument('--cloned_speaker_name', type=str, default='speaker')
    
    # clone voice from npz
    parser.add_argument('--text_prompts', type=str, default='I love you guys')
    parser.add_argument('--speaker_name', type=str, default='speaker')
    parser.add_argument('--saved_models', type=str, default='./models')
    parser.add_argument('--text_temp', type=float, default=0.7)
    parser.add_argument('--waveform_temp', type=float, default=0.7)
    parser.add_argument('--cloned_output',type=str, default='./cloned_output.wav')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args

def print_args(args):
    pass