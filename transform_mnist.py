import argparse, sys
import torch, torchvision
import pandas as pd
import numpy as np

# Note: Much of this code is thanks to code associated with the colored MNIST experiment
# from Invariant Risk Minimization
# https://github.com/facebookresearch/InvariantRiskMinimization/blob/master/code/colored_mnist/main.py


def make_environment(X, y, e):
    # Take the data sources to be the digit labels
    sources = y
    # Assign binary label based on digit
    y = (y < 5)
    # Flip label with probability 1/4
    y = y.__xor__(torch.rand(len(y)) < 0.25)
    # Assign color based on label, and flip w.p. e
    colors = y.__xor__(torch.rand(len(y)) < e)

    # 4x subsample pixels to ease computational burden
    X = X[:, ::4, ::4]
    # Create the color channels
    X = torch.stack([X, X], dim=1)
    # Zero out the second channel of pixels when color is set to 1
    X[:, 1, :, :] *= (1-colors)[:, None, None]
    # Reshape dataset so each datapoint is a single row
    X = X.flatten(start_dim=1)

    return X, sources, y


def transform(args):
    print("in transform")
    mnist_train = torchvision.datasets.MNIST(root=args.dataset_path, train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(root=args.dataset_path, train=False, download=True)

    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train.data.numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train.targets.numpy())

    rng_state = np.random.get_state()
    np.random.shuffle(mnist_test.data.numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_test.targets.numpy())

    train_mode, val_mode, test_mode = args.mode.split("|")
    train_color_flip_probs = [float(e) for e in train_mode.split(":")[1].split(",")]
    val_color_flip_prob = float(val_mode.split(":")[1])
    test_color_flip_prob = float(test_mode.split(":")[1])
    # In training, color has positive correlation with class labels
    envs = {
        'train': {
            'e': [],
            'X': [],
            'src': [],
            'y': []
        }
    }
    for idx_e, train_e in enumerate(train_color_flip_probs):
        X_train, source_train, y_train = make_environment(X=mnist_train.data[idx_e::len(train_color_flip_probs)],
                                                          y=mnist_train.targets[idx_e::len(train_color_flip_probs)],
                                                          e=train_e)
        envs['train']['e'].append(train_e)
        envs['train']['X'].append(X_train)
        envs['train']['src'].append(source_train)
        envs['train']['y'].append(y_train)
    X_validate, sources_validate, y_validate = make_environment(mnist_test.data[:1000], mnist_test.targets[:1000], val_color_flip_prob)
    X_test, sources_test, y_test = make_environment(mnist_test.data[1000:], mnist_test.targets[1000:], test_color_flip_prob)

    X = torch.cat(envs['train']['X'], dim=0)
    X = torch.cat([X, X_validate, X_test], dim=0)
    sources = torch.cat(envs['train']['src'], dim=0)
    sources = torch.cat([sources, sources_validate, sources_test], dim=0)
    y = torch.cat(envs['train']['y'], dim=0)
    y = torch.cat([y, y_validate, y_test], dim=0)
    train_val_test_split = np.concatenate([[1]*len(mnist_train.data), [2]*len(X_validate), [3]*len(X_test)])
    environments = np.concatenate([ [str(envs['train']['e'][i])]*len(envs['train']['X'][i]) for i in range(len(train_color_flip_probs)) ])
    environments = np.concatenate([environments, [str(val_color_flip_prob)]*len(X_validate), [str(test_color_flip_prob)]*len(X_test)])

    df = pd.DataFrame(X.numpy(), columns=["{:03d}".format(i) for i in range(X.shape[1])])
    df['digit_num'] = sources.numpy()
    df['environment'] = environments
    df['train_val_test_split'] = train_val_test_split
    df['smaller_than_five'] = y

    print("Transformed columns:")
    print(df.columns)

    print("Outputting to CSV")
    outfile = args.output_dir + "/" + args.output_file + "_" + args.mode + ".csv"
    df.to_csv(outfile, index=False)


def main():
    print("in main")
    parser = argparse.ArgumentParser(description="Process the dataset")
    parser.add_argument("--dataset-path", type=str, required=True, help="path to CSV to process")
    parser.add_argument("--dataset-id", type=str, required=True, choices=['mnist'], help="the dataset identifier to process")
    parser.add_argument("--mode", type=str, default="", help="Specification of training and testing environment color flip probabilites of form 'train:<pr1>,<pr2>,...,<prn>|test:<pr>'")
    parser.add_argument("--output-dir", type=str, required=True, help="The directory in which output files will be placed")
    parser.add_argument("--output-file", type=str, required=True, help="File to write the output to")

    args = parser.parse_args()
    print(args)
    transform(args)


if __name__ == "__main__":
    # sys.argv.extend([
    #     "--dataset-path", "./derp/",
    #     "--dataset-id", "mnist",
    #     "--mode", "train:0.1,0.2,0.7|val:0.9|test:0.5",
    #     "--output-dir", "./derp",
    #     "--output-file", "transformed"
    # ])
    main()
