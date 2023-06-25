import argparse
import torch
from pathlib import Path

def get_args():
    # create parser args
    parser = argparse.ArgumentParser()

    # senario
    parser.add_argument('--scenario', type=str, default='train')

    args = parser.parse_args()
    return args

def print_args(args):
    pass