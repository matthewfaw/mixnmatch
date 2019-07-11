# Mix&Match: An Optimistic Tree-Search Approach for Learning Models from Mixture Distributions

This is the code used to run simulations associated with our Mix&Match paper.

The main entrypoint to run an experiment is:
`run_single_experiment.py`. This script can be run locally,
or on k8s using one of the scripts in `experiment_running` folder.

The project assumes the following environment variables have been set:
- `GCLOUD_DATASET_BUCKET_BASE`: The base bucket name where datasets and experiment results will be stored. E.g. `gs://your-bucket-name`
- `GCLOUD_PROJECT`: The project name associated with all Google cloud resources you're using. Note that this can be different from your google cloud project name.

The infrastructure for running all code in this project can be created by
following the setup instructions [here](https://github.com/matthewfaw/mixnmatch-infrastructure).

Running this python script with no arguments will describe the argument
settings available. The python requirements are provided in requirements.txt.

To run the allstate dataset, first preprocess the dataset using the scripts in the
`dataset_creation` folder,
`transform_allstate.sh` (to preprocess the dataset), then `create_allstate_*.sh`
 (to partition the dataset in the desired manner),
then run one of the allstate run all scripts in the `experiment_running`
folder.
 The wine dataset can be created by running the same sequence of 
the corresponding scripts.  The mnist experiment can be run using the above
steps, except skipping the `transform_...sh` script. 
Note that the `transform_*sh` script runs the `transform_*.py` file, and the
`create_*.sh` script runs the `process.py` file.

The Docker container packaging the necessary scripts can be created
either by running the `./build_docker.sh` script, or by setting up a Jenkins
project and running the `Jenkinsfile`. Note that the `project` variable
in the Jenkinsfile should be changed to point to your Google Cloud project id.

After the container has been published, the experiments can be run using one
of the scripts in the `experiment_running` folder.  Hyperparameter tuning
is performed using [Katib](https://github.com/kubeflow/katib) by running the
`katib_experiment_*.sh` scripts,
and testing experiments can be run using `<DATASET_ID>_run_all.sh`.

Experiments can be visualized by using the code in `ExamineGraphs.ipynb`.
A convenience script (and instructions for running the script) that can be used to spin up a Jupyter server 
and run this notebook can be found
[here](https://github.com/matthewfaw/mixnmatch-infrastructure).

The datasets currently supported are:
- [Allstate](https://www.kaggle.com/c/allstate-purchase-prediction-challenge/data)
- [Wine](https://www.kaggle.com/dbahri/wine-ratings)
- [MNIST](https://pytorch.org/docs/stable/_modules/torchvision/datasets/mnist.html)

Code dealing with data lives in the `datasets/` dir, and code dealing with
experiment running, and defining the tree search data structures, lives in
the `mf_tree/` dir.

More documentation to individual files will be added in the future.

I have run these scripts on a Macbook Pro with 
2.7 GHz Intel Core i7 and 16GB of RAM to create the experiments in a kubernetes cluster
(created by [this](https://github.com/matthewfaw/mixnmatch-infrastructure) project)
running in Google Cloud.