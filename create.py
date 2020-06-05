import argparse, sys
import dill as pickle
from datasets.pandas_dataset import PandasData, SparseOHEPandasData


def process(args):
    breakdown = {}
    vals_to_split = []
    for entry in args.val_train_val_test_drop.split('|'):
        val, tvt = entry.split(':')
        train, validate, test, drop = [float(v) for v in tvt.split(',')]
        if sum([train, validate, test, drop]) <= 1.0:
            setting = "percents"
        else:
            setting = "total"
        print("Determined train/val/test/drop split reported as", setting)
        breakdown[val] = {
            "setting": setting,
            "train": float(train),
            "validate": float(validate),
            "test": float(test),
            "drop": float(drop)
        }
        vals_to_split.append(val)

    print("Creating dataset object")
    if args.dataset_id in ["allstate", "wine", "mnist"]:
        dataset = PandasData(csv_file=args.dataset_path,
                             dataset_id=args.dataset_id,
                             product_key_to_keep=args.target,
                             is_categorical=args.is_categorical,
                             key_to_split_on=args.split_key,
                             vals_to_split=vals_to_split,
                             col_to_filter=args.col_to_filter,
                             vals_to_keep_in_filtered_col=args.vals_to_keep_in_filtered_col.split(','),
                             cols_to_drop=args.cols_to_drop.split(','),
                             breakdown=breakdown)
    elif args.dataset_id in ["amazon"]:
        dataset = SparseOHEPandasData(csv_file=args.dataset_path,
                                      dataset_id=args.dataset_id,
                                      product_key_to_keep=args.target,
                                      is_categorical=args.is_categorical,
                                      key_to_split_on=args.split_key,
                                      vals_to_split=vals_to_split,
                                      col_to_filter=args.col_to_filter,
                                      vals_to_keep_in_filtered_col=args.vals_to_keep_in_filtered_col.split(','),
                                      cols_to_drop=args.cols_to_drop.split(','),
                                      breakdown=breakdown)
    else:
        print("Unsupported dataset id",args.dataset_id)
        assert False

    filename = "{}/{}_{}_{}_{}_{}.p".format(args.output_dir,
                                            args.dataset_id,
                                            args.split_key,
                                            args.target,
                                            args.val_train_val_test_drop,
                                            args.unique_image_tag)

    print("Dumping dataset object to", filename)
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)


def main():
    parser = argparse.ArgumentParser(description="Process the dataset")
    parser.add_argument("--dataset-path", type=str, required=True, help="path to CSV to process if pandas dataset. Otherwise, path to location to store the downloaded PyTorch dataset")
    parser.add_argument("--dataset-id", type=str, required=True, choices=['allstate', 'wine', 'amazon', 'mnist'], help="the dataset identifier to process")
    parser.add_argument("--is-categorical", type=bool, required=True, help="indicates whether or not the dataset has categorical targets. If _any_ string is passed to this arg, then this will be true. If an empty string is passed, then it will be false")
    parser.add_argument("--split-key", type=str, required=True, help="Key to split the dataset on")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--cols-to-drop", type=str, default="", help="Column names to drop")
    parser.add_argument("--col-to-filter", type=str, default="", help="The column to filter. For example, in the wine experiment, if the split key is country, you can use this setting to filter out only data in particular price quartiles.")
    parser.add_argument("--vals-to-keep-in-filtered-col", type=str, default="", help="The comma-separated list of values to keep in the filtered column")
    parser.add_argument("--val-train-val-test-drop", type=str, required=True, help="State1:Train,Validate,Test,Drop|... percentage split")
    parser.add_argument("--output-dir", type=str, required=True, help="The directory in which output files will be placed")
    parser.add_argument("--unique-image-tag", type=str, required=True, help="The tag of the image currently being used -- intendent to be the tag that doesn't change")

    args = parser.parse_args()
    print(args)
    process(args)


if __name__ == "__main__":
    # sys.argv.extend([
    #     "--dataset-path","./derp/transformed_train:0.1,0.2,0.7|test:0.9.csv",
    #     "--dataset-id", 'mnist',
    #     "--is-categorical", "True",
    #     "--split-key", "environment",
    #     "--target", "smaller_than_five",
    #     "--cols-to-drop", "",
    #     "--col-to-filter", "",
    #     "--vals-to-keep-in-filtered-col", "",
    #     "--val-train-val-test-drop", "0.1:1,0,0,0|0.2:1,0,0,0|0.7:1,0,0,0|0.9:0,0.3,0.7,0",
    #     "--output-dir", "./derp",
    #     "--unique-image-tag", "latest"
    # ])
    main()
