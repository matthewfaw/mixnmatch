from configuration.output_file_configuration import setup_output_file

from pprint import pprint

from configuration.dataset_configuration import DatasetConfiguration
from configuration.experiment_configuration import ExperimentConfiguration
from configuration.mmd_configuration import MMDConfiguration
from configuration.model_configuration import ModelConfiguration
from configuration.optimization_configuration import OptimizationConfiguration
from configuration.tree_search_configuration import TreeSearchConfiguration
from mf_tree.experiment_runner import ExperimentRunner


class DefaultExperimentConfigurer:
    def __init__(self,
                 dataset_config: DatasetConfiguration,
                 experiment_config: ExperimentConfiguration,
                 model_config: ModelConfiguration,
                 optimization_config: OptimizationConfiguration,
                 mmd_config: MMDConfiguration,
                 tree_search_config: TreeSearchConfiguration,
                 output_dir,
                 tag):
        self.dataset_config = dataset_config
        self.experiment_config = experiment_config
        self.model_config = model_config
        self.optimization_config = optimization_config
        self.mmd_config = mmd_config
        self.tree_search_config = tree_search_config

        self.output_dir = output_dir
        self.output_filename = setup_output_file(dataset_config=dataset_config,
                                                 experiment_config=experiment_config,
                                                 model_config=model_config,
                                                 optimization_config=optimization_config,
                                                 mmd_config=mmd_config,
                                                 tree_search_config=tree_search_config,
                                                 tag=tag)

    def configure(self):
        self.dataset_config.configure(optimization_config=self.optimization_config,
                                      mmd_config=self.mmd_config)
        self.experiment_config.configure(data=self.dataset_config.data)
        self.model_config.configure(data=self.dataset_config.data, eta=self.optimization_config.eta)

        print("Dataset config:")
        pprint(vars(self.dataset_config))
        print("Experiment config:")
        pprint(vars(self.experiment_config))
        print("Model config:")
        pprint(vars(self.model_config))
        print("Optimization config:")
        pprint(vars(self.optimization_config))
        print("MMD config:")
        pprint(vars(self.mmd_config)) if self.mmd_config is not None else print(self.mmd_config)
        print("Tree search config:")
        pprint(vars(self.tree_search_config))
        print("Output dir:", self.output_dir)
        print("Output file:", self.output_filename)

        return ExperimentRunner(dataset_config=self.dataset_config,
                                experiment_config=self.experiment_config,
                                model_config=self.model_config,
                                optimization_config=self.optimization_config,
                                mmd_config=self.mmd_config,
                                tree_search_config=self.tree_search_config,
                                output_dir=self.output_dir,
                                output_file=self.output_filename)
