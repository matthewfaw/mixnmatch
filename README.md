# Mix&Match: An Optimistic Tree-Search Approach for Learning Models from Mixture Distributions

**This is the code used to run simulations associated with our Mix&Match paper:**
 
 _Please cite the above paper if this code is used in any publication._
 
 **Code to setup the infrastructure to run these experiments in Google Cloud can be found in the other project folder.** 

## tl;dr
### Entrypoints for experiments in paper

The main entrypoint to run an experiment is:
`run_single_experiment.py`. This script can be run locally,
or on k8s using one of the scripts in `experiment_running` folder. 
The scripts in the `experiment_running` folder demonstrate exactly
how the experiments were run to generate results for the paper.
Here is how the scripts correspond to our experiments/datasets:
- `experiment_running/allstate_aimk_alt_newfeats_alt2.sh`: Allstate experiment corresponding to Figures 1a&2, and corresponding to dataset `neurips_datasets/allstate_fig1aand2.p`
- `experiment_running/wine_new_country_alt11_no_price_quantiles.sh`: Wine experiment corresponding to Figure 3, and corresponding to dataset `neurips_datasets/wine_fig3.p`
- `experiment_running/wine_ppq_even_smaller_us.sh`: Wine experiment corresponding to Figure 4, and corresponding to dataset `neurips_datasets/wine_fig4.p`
- `experiment_running/mnist.p`: Amazon experiment corresponding to Figure 5, and corresponding to dataset `neurips_datasets/mnist_fig5.p`
- `experiment_running/mnist_diff_test.p`: Amazon experiment corresponding to Figure 6, and corresponding to dataset `neurips_datasets/mnist_fig6.p`
- `experiment_running/mnist_slight_shift.p`: Amazon experiment corresponding to Figure 1b and 7, and corresponding to dataset `neurips_datasets/mnist_fig1band7.p`

### Supported datasets
The datasets currently supported are:
- [Allstate](https://www.kaggle.com/c/allstate-purchase-prediction-challenge/data)
- [Wine](https://www.kaggle.com/dbahri/wine-ratings)
- [MNIST](https://pytorch.org/cppdocs/api/classtorch_1_1data_1_1datasets_1_1_m_n_i_s_t.html)

### Testing environments
I have run these scripts on a Macbook Pro with 
2.7 GHz Intel Core i7 and 16GB of RAM to create the experiments in a Kubernetes cluster
(created by the other included project)
running in Google Cloud using n1-standard-4 servers with 40GB of disk space.

## The details
### Necessary environment setup
If you would like to run the scripts as I have them set up (i.e., run them in a Kubernetes cluster),
then you'll need to have the following environment variables have been set:
- `GCLOUD_DATASET_BUCKET`: The base bucket name where datasets and experiment results will be stored. E.g. `gs://your-bucket-name`
- `GCLOUD_PROJECT`: The project name associated with all Google cloud resources you're using. Note that this can be different from your google cloud project name.

The infrastructure for running all code in this project can be created by
following the setup instructions in the other project folder.

Running this python script with no arguments will describe the argument
settings available. The python requirements are provided in `requirements.txt`.
Note that I use `python:3.7.3` for Docker containers for running these scripts, so if you're running
locally, you should also be using `python3`. 

### Creating a new dataset
The codebase is designed to easily create a new custom dataset from the 
original Kaggle datasets (e.g., make a custom split, transform features, ect.).
The following is not necessary if you simply want to use the datasets from the 
`neurips_datasets` folder. However, if you'd like to make a new dataset, 
follow these instructions.

To create a new dataset (e.g., an allstate dataset), take the following steps:
- Run one of the `transform_...` scripts in the `/dataset_transformation` folder. 
This script will download the dataset from the original source (e.g., Kaggle), 
and perform the feature transformation used for the experiments in our paper.
It will not, however, split the dataset into K sources.
Note that these scripts call the `transform_<DATA SOURCE>.py` python script. You can also just run this script directly.
- Use the output dataset `csv` of the `transform_...` script as an input to the
dataset creation scripts, which can be found in `dataset_creation` folder, each
named like `create_...`. This script will split the dataset in the desired
proportions and pickle the dataset in a format ready to use by the 
`experiment_running` scripts.
Note that these scripts all call the `create.py` python script. You can also just run this script directly
- Optionally, use the output of a `dataset_creation` script as an input to the `dataset_postprocessing` script. 
Currently, the only functionality of these scripts is to compute importance weights for the training data to be used
downstream for importance-weighted ERM

### Hyperparameter tuning
Hyperparameter tuning was performed in the Katib framework, using scripts in
the `hyperparameter_tuning` folder. Running these scripts will require having 
a running Kubernetes cluster, or simply running the python scripts that these scripts invoke.

Note that these scripts also call the `run_single_experiment.py` script, using the validation loss instead of test loss 
as the metric to report

### Creating and using Docker containers for the runtime environment
All experiments for this paper were run inside Docker containers so that the 
experiments can easily be run by others.
The Docker container packaging the code from this repo to run experiments 
and create datasets can be created
either by running the `./build_docker.sh` script, or by setting up a Jenkins
project and running the `Jenkinsfile`. Note that the `project` variable
in the Jenkinsfile should be changed to point to your Google Cloud project id.

After the container has been published (to the Google container registry),
the experiments can be run using one
of the scripts in the `experiment_running` folder.  Hyperparameter tuning
is performed using [Katib](https://github.com/kubeflow/katib) by running the
`katib_experiment_*.sh` scripts,
and testing experiments can be run using an `experiment_running` script.

### Viewing Experiment Results
Experiments can be visualized by using the code in `ExamineGraphs.ipynb`.
A convenience script (and instructions for running the script) that can 
be used to spin up a Jupyter server 
and run this notebook can be found in the other project folder.

#### Experiment Results in the paper
The outputs of running the experiments in the paper are saved in
``datasets_in_paper/experiment_running``.
These are the results that can be loaded into a Jupyter notebook (``ExamineGraphs.ipynb`) 
and generate all of the plots. Note that there are scripts in the sister infrastructure project
to generate a Jupyter notebook with these files.

Also, note that the ``datasets_in_paper/transformed`` folder contains all of the pre-processed 
datasets used for the paper, before they were split.


## Python Code layout
Code dealing with data lives in the `datasets/` dir, and code dealing with
experiment running, and defining the tree search data structures, lives in
the `mf_tree/` dir.

Entrypoint code is in the base dir of this project.

## License

**This project is licensed under the terms of the Apache 2.0 License.**
