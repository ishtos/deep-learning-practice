import os 
import sys
import argparse
import cv2

def main():
    pass


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', type=bool, default=False)
    
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))