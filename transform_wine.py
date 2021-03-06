import argparse, os, sys
import pandas as pd
import numpy as np


# def remove_some_classes(df):
#     print("Removing some classes")
#     classes_to_keep = [
#         'acid', 'angular', 'austere', 'barnyard', 'bright', 'butter', 'cassis',
#         'charcoal', 'cigar', 'complex', 'cream', 'crisp', 'dense', 'earth',
#         'elegant', 'flabby', 'flamboyant', 'fleshy', 'food friendly', 'grip',
#         'hint of', 'intellectual', 'jam', 'juicy', 'laser', 'lees', 'mineral',
#         'oak', 'opulent', 'refined', 'silk', 'steel', 'structure', 'tannin',
#         'tight', 'toast', 'unctuous', 'unoaked', 'velvet', 'points', 'price'
#     ]
#     return df[classes_to_keep]


def collapse_countries(df):
    print("Undo one hot encoding on countries")
    countries = ['Argentina', 'Australia', 'Austria', 'Bulgaria', 'Canada',
                 'Chile', 'France', 'Germany', 'Greece', 'Hungary', 'Israel', 'Italy',
                 'New Zealand', 'Portugal', 'Romania', 'South Africa', 'Spain', 'Turkey',
                 'US', 'Uruguay', 'Other']
    df['country'] = np.NaN
    for country in countries:
        df.loc[df[country] == 1, 'country'] = country
    df.dropna(inplace=True)
    return df.drop(columns=countries)


def add_price_quantiles(df):
    print("Adding price quantiles")
    prices = df['price']
    df['price_quartile'] = ["q1" if p <= 17 else ("q2" if p <= 25 else ("q3" if p <= 42 else "q4")) for p in prices]
    return df


def log_transform_price(df):
    print("Log transforming price")
    df.loc[:,'price'] = np.log(df['price'])
    return df


def transform(args):
    print("in transform")
    train_name, val_name, test_name = args.dataset_path.split(',')
    dirname = os.path.dirname(train_name)
    val_name = dirname + "/" + val_name
    test_name = dirname + "/" + test_name
    print("Loading train")
    df_train = pd.read_csv(train_name, index_col=0)
    print("Loading validate")
    df_validate = pd.read_csv(val_name,index_col=0)
    print("Loading test")
    df_test = pd.read_csv(test_name,index_col=0)

    print("Combining datasets")
    df = pd.concat((df_train, df_validate, df_test))

    df_transformed = log_transform_price(add_price_quantiles(df))
    if args.mode == "collapse-countries":
        print("Determined that countries should be collapsed:")
        df_transformed = collapse_countries(df_transformed)
    else:
        print("{} is not a valid mode... Skipping".format(args.mode))

    print("Transformed columns:")
    print(df_transformed.columns)

    print("Outputting to CSV")
    df_transformed.to_csv(args.output_dir + "/" + args.output_file)


def main():
    print("in main")
    parser = argparse.ArgumentParser(description="Process the dataset")
    parser.add_argument("--dataset-path", type=str, required=True, help="path to CSV to process")
    parser.add_argument("--dataset-id", type=str, required=True, choices=['wine'], help="the dataset identifier to process")
    parser.add_argument("--mode", type=str, default="", choices=["", "collapse-countries"], help="The mode determining whether or not to collapse one-hot encoded countries into a single column")
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
