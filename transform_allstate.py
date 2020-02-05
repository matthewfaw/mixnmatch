import argparse, sys
import pandas as pd


def _flatten(df):
    print("in flatten:")

    print("Preprocess some cols")
    df["risk_factor"] = df["risk_factor"].fillna(-1)

    print("Get df for intermetidate vals")
    df_intermediate_choices = df.loc[df["record_type"] == 0].copy()
    df_intermediate_choices["day_first"] = df_intermediate_choices["day"]

    print("Get df for final vals")
    df_final_choice = df.loc[df['record_type'] == 1].copy()
    df_final_choice = df_final_choice.drop(["record_type", "time", "location"], 1)

    final_cols_renaming = {
        chr(i) : chr(i) + "_final" for i in range(ord("A"), ord("G")+1)
    }
    final_cols_renaming["cost"] = "cost_final"
    df_final_choice.rename(columns=final_cols_renaming, inplace=True)

    print("aggregate intermediate vals:")
    aggregate_ops = {
        "customer_ID": "first",
        "day_first": "first",
        "day": "last",
        "A": "last",
        "B": "last",
        "C": "last",
        "D": "last",
        "E": "last",
        "F": "last",
        "G": "last",
        "cost": "last",
    }
    df_intermediate_choices = df_intermediate_choices.groupby(df_intermediate_choices['customer_ID']).agg(aggregate_ops)
    df_intermediate_choices.rename(columns={
        "day_first": "day_first_prev",
        "day": "day_prev",
        "A": "A_prev",
        "B": "B_prev",
        "C": "C_prev",
        "D": "D_prev",
        "E": "E_prev",
        "F": "F_prev",
        "G": "G_prev",
        "cost": "cost_prev",
    }, inplace=True)

    print("Join intermediate and final results")
    df_joined = df_intermediate_choices.set_index('customer_ID').join(df_final_choice.set_index('customer_ID'))

    print("Create some new cols based on intermediate + final features")
    df_joined["num_days"] = (df_joined["day"] - df_joined["day_first_prev"]) % 7
    df_joined["final_purchase_date_same"] = (df_joined["day"] == df_joined["day_prev"]).astype(int)

    df_joined["cost_changed"] = (df_joined["cost_final"] != df_joined["cost_prev"]).astype(int)
    df_joined["cost_delta"] = df_joined["cost_final"] - df_joined["cost_prev"]

    print("Drop some tmp columns")
    df_joined = df_joined.drop(['day_first_prev', 'day_prev', "cost_prev"], 1)

    return df_joined


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
