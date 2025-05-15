import pandas as pd
import argparse
import pickle

parser = argparse.ArgumentParser(description="Read and print a pickle file.")
parser.add_argument('file', type=str, help='Path to the pickle file')
args = parser.parse_args()


with open(args.file, "rb") as f:
    data = pickle.load(f)
    for item in data:
        print(item)
