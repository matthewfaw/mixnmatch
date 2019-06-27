import argparse
import pandas as pd
import numpy as np
from datetime import datetime as dt


def _flatten(df):
    print("First, collect dataframe of final customer results")
    a_g_list = range(ord('A'), ord('G')+1)
    final_cols_to_keep = [chr(i) for i in a_g_list]
    final_cols_to_keep.extend(['customer_ID', 'cost'])

    df_final_choice = df.loc[df["record_type"] == 1].drop(df.columns.difference(final_cols_to_keep), 1)

    final_cols_renaming = {
        chr(i) : chr(i) + "_final" for i in a_g_list
    }
    final_cols_renaming["cost"] = "cost_final"
    df_final_choice.rename(columns=final_cols_renaming, inplace=True)

    print("Add final cust results as new columns to original df")
    df = df.set_index('customer_ID').join(df_final_choice.set_index('customer_ID'))
    df = df.reset_index()

    print("Add a column to keep track of number of datapoints for each cust id")
    print("We'll sum this to count in the agg step")
    df["num_shopping_pts"] = np.ones((df.shape[0],)).astype(int)
    print("Preprocess some cols")
    df["risk_factor"] = df["risk_factor"].fillna(-1)
    df["time"] = df["time"].apply(lambda t: dt.strptime(t, "%H:%M"))

    datetime_lambda = lambda ts: (max(ts) - min(ts)).seconds
    print("aggregate:")
    aggregate_ops = {
        "num_shopping_pts": "sum",
        "day": np.ptp,
        "time": datetime_lambda,
        "state": "first",
        "location": "first",
        "group_size": "first",
        "homeowner": "first",
        "car_age": "first",
        "car_value": "first",
        "risk_factor": "first",
        "age_oldest": "first",
        "age_youngest": "first",
        "married_couple": "first",
        "C_previous": "first",
        "duration_previous": "first",
        "A": np.ptp,
        "B": np.ptp,
        "C": np.ptp,
        "D": np.ptp,
        "E": np.ptp,
        "F": np.ptp,
        "G": np.ptp,
        "cost": "mean",
        "A_final": "first",
        "B_final": "first",
        "C_final": "first",
        "D_final": "first",
        "E_final": "first",
        "F_final": "first",
        "G_final": "first",
        "cost_final": "first",
    }
    df = df.groupby(df['customer_ID']).agg(aggregate_ops)
    print("Rename some of the cols that have been manipulated")
    df.rename(columns={
        "day": "num_days",
        "time": "time_range_sec",
        "A": "A_changes",
        "B": "B_changes",
        "C": "C_changes",
        "D": "D_changes",
        "E": "E_changes",
        "F": "F_changes",
        "G": "G_changes",
        "cost": "cost_avg",
    }, inplace=True)
    return df


def _ohe(df):
    print("in ohe")
    return pd.get_dummies(df, columns=['car_value'])


def transform(args):
    print("in transform")
    df = pd.read_csv(args.dataset_path)

    df_processed = _ohe(_flatten(df))

    df_processed.to_csv(args.output_dir + "/" + args.output_file)


def main():
    print("in main")
    parser = argparse.ArgumentParser(description="Process the dataset")
    parser.add_argument("--dataset-path", type=str, required=True, help="path to CSV to process")
    parser.add_argument("--dataset-id", type=str, required=True, choices=['allstate'], help="the dataset identifier to process")
    parser.add_argument("--output-dir", type=str, required=True, help="The directory in which output files will be placed")
    parser.add_argument("--output-file", type=str, required=True, help="File to write the output to")

    args = parser.parse_args()
    print(args)
    transform(args)


if __name__ == "__main__":
    main()
