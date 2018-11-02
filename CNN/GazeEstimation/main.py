import os
import sys
import argparse
import numpy as np
import pandas as pd
from pytorch_model import GazeNet

def main(args):
    model = GazeNet()
    print(model)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.01)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))