import argparse, os, sys
import pandas as pd
import numpy as np


def transform(args):
    print("in transform")
    df = pd.read_csv(args.dataset_path, dtype=np.str)

    df_processed = df

    df_processed.to_csv(args.output_dir + "/" + args.output_file, index=False)


def main():
    print("in main")
    parser = argparse.ArgumentParser(description="Process the dataset")
    parser.add_argument("--dataset-path", type=str, required=True, help="path to CSV to process")
    parser.add_argument("--dataset-id", type=str, required=True, choices=['amazon'], help="the dataset identifier to process")
    parser.add_argument("--mode", type=str, default="", choices=[""], help="Currently has no function")
    parser.add_argument("--output-dir", type=str, required=True, help="The directory in which output files will be placed")
    parser.add_argument("--output-file", type=str, required=True, help="File to write the output to")

    args = parser.parse_args()
    print(args)
    transform(args)

if __name__ == "__main__":
    # sys.argv.extend([
    #     "--dataset-path", "derp/train.csv",
    #     "--dataset-id", "allstate",
    #     "--output-dir", "outderp",
    #     "--output-file", "derp"
    # ])
    main()
