import argparse, sys, os
import dill as pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from datasets.pandas_dataset import PandasData
import numpy as np


BIG_NUM = 1000


def _get_output_file(args):
    input_file = os.path.basename(args.dataset_path)
    input_file_no_ext = os.path.splitext(input_file)[0]
    return args.output_dir + "/" + input_file_no_ext + "_iw.p"


def _split_training_and_validation_data(data: PandasData):
    print('Split training and validation data in half -- half for importance weighting, half for training')
    print("Originally, training data is size: ", data.train.shape)
    print("Originally, validation data is size: ", data.validate.shape)
    print("Originally, test data is size: ", data.test.shape)
    train_s1 = None
    train_s2 = None
    validate_s1 = None
    validate_s2 = None

    for split_val in data.vals_to_split:
        train_at_val = data.train.loc[data.train[data.key_to_split_on] == split_val]
        validate_at_val = data.validate.loc[data.validate[data.key_to_split_on] == split_val]

        if len(train_at_val) > 0:
            train_at_val_s1, train_at_val_s2 = train_test_split(train_at_val, train_size=0.5)

            if train_s1 is None:
                train_s1, train_s2 = train_at_val_s1, train_at_val_s2
            else:
                train_s1 = pd.concat((train_s1, train_at_val_s1))
                train_s2 = pd.concat((train_s2, train_at_val_s2))
        else:
            print("Skipping splitting training data for {} since no data in this split".format(split_val))

        if len(validate_at_val) > 0:
            validate_at_val_s1, validate_at_val_s2 = train_test_split(validate_at_val, train_size=0.5)
            if validate_s1 is None:
                validate_s1, validate_s2 = validate_at_val_s1, validate_at_val_s2
            else:
                validate_s1 = pd.concat((validate_s1, validate_at_val_s1))
                validate_s2 = pd.concat((validate_s2, validate_at_val_s2))
        else:
            print("Skipping splitting validation data for {} since no data in this split".format(split_val))
    print("After splitting...")
    print('Training split 1 size:', train_s1.shape, 'Training split 2 size:', train_s2.shape)
    print('Validate split 1 size:', validate_s1.shape, 'Validate split 2 size:', validate_s2.shape)

    return train_s1, train_s2, validate_s1, validate_s2


def _importance_weight(data: PandasData, max_lr_iters):
    train_s1, train_s2, validate_s1, validate_s2 = _split_training_and_validation_data(data=data)

    print('Adding indicator variables is_validate to first split')
    train_s1['is_validate'] = 0
    validate_s1['is_validate'] = 1
    print('Training logistic regression model on train+validation split 1 to generate importance weights')
    train_val_s1 = pd.concat((train_s1,validate_s1))
    x_cols = sorted(list(set(train_val_s1.columns) - {data.product_key_to_keep, 'is_validate'}))
    X = train_val_s1[x_cols]
    X = pd.get_dummies(X, columns=[data.key_to_split_on])
    X.fillna(0)
    y = train_val_s1['is_validate']
    print('LogisticRegression trained with features:',X.columns)
    is_validate_model = LogisticRegression(max_iter=max_lr_iters).fit(X, y)

    print('Compute importance weights on training split 2')
    X_s2 = train_s2[x_cols]
    X_s2 = pd.get_dummies(X_s2, columns=[data.key_to_split_on])
    for missing_col in set(X.columns) - set(X_s2.columns):
        print("Filling missing column {} with 0".format(missing_col))
        X_s2[missing_col]=0
    pr = is_validate_model.predict_proba(X_s2)
    importance_weights = pr[:, 1]/pr[:, 0]
    importance_weights[np.isnan(importance_weights)] = 0.
    importance_weights[np.isinf(importance_weights)] = BIG_NUM
    importance_weights /= max(importance_weights)
    print("Importance weights: mean: {:.3f}, var: {:.3f}, min: {:.3f}, max: {:.3f}".format(np.mean(importance_weights),
                                                                                           np.var(importance_weights),
                                                                                           np.min(importance_weights),
                                                                                           np.max(importance_weights)))

    print('Create importance weight column for train, validate, test')
    data.importance_weight_column_name = 'importance_weights'

    # Append exp(logit()) score to each predicted sample
    train_s2[data.importance_weight_column_name] = importance_weights
    # Append 1's to validation + test
    validate_s2[data.importance_weight_column_name] = 1
    data.test[data.importance_weight_column_name] = 1

    print('Update the data in data.train/validate/test')
    column_names = x_cols.copy()
    column_names.extend(['importance_weights', data.product_key_to_keep])
    print('Train/Validate/test will have column names:', column_names)

    data.train = train_s2[column_names]
    data.validate = validate_s2[column_names]
    data.test = data.test[column_names]

    print('After updating, we have')
    print("data.train.shape: ", data.train.shape)
    print("data.validate.shape: ", data.validate.shape)
    print("data.test.shape: ", data.test.shape)

    return data


def process(args):
    with open(args.dataset_path, 'rb') as f:
        data = pickle.load(f)

    if args.postprocessing_step == "importance-weight":
        print("Beginning importance weight postprocessing now")
        out_data = _importance_weight(data, args.max_lr_iters)
    else:
        print("Postprocessing step for " + args.postprocessing_step + " not configured. Cannot continue")
        assert False

    output_file = _get_output_file(args)
    print("Dumping to output filename:", output_file)
    with open(output_file, "wb") as f:
        pickle.dump(out_data, f)


def main():
    parser = argparse.ArgumentParser(description="Run a single experiment")
    parser.add_argument("--dataset-path", type=str, required=True, help="path to pickled _Data object to process")
    parser.add_argument("--dataset-id", type=str, required=True, choices=['allstate', 'wine', 'amazon', 'mnist'], help="the dataset identifier to process")
    parser.add_argument("--postprocessing-step", type=str, required=True, choices=['importance-weight'], help="the type of postprocessing to perform")
    parser.add_argument("--max-lr-iters", type=int, required=True, help="number of optimization steps for logistic regression solver")
    parser.add_argument("--output-dir", type=str, required=True, help="The directory in which output files will be placed")
    parser.add_argument("--tag", type=str, default="missingtag", help="The image tag used in running this experiment")

    args = parser.parse_args()
    print(args)
    process(args)


if __name__ == "__main__":
    # sys.argv.extend([
    #     "--dataset-path", "derp/mnist_environment_smaller_than_five_0.1:1,0,0,0|0.2:1,0,0,0|0.7:1,0,0,0|0.9:0,0.3,0.7,0_latest.p",
    #     "--dataset-id", "mnist",
    #     "--postprocessing-step", "importance-weight",
    #     "--max-lr-iters", "5000",
    #     "--tag", "latest",
    #     "--output-dir", "out"
    # ])
    main()