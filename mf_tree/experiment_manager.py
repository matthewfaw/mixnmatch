from mf_tree.experiment_configurer import DefaultExperimentConfigurer

from datetime import datetime as dt
import dill as pickle


class ExperimentManager:
    def __init__(self, experiment_configurer: DefaultExperimentConfigurer):
        self.experiment_configurer = experiment_configurer
        self.experiment_config = experiment_configurer.experiment_config
        self.experiment_file_prefix = experiment_configurer.output_dir + "/" + experiment_configurer.output_filename
        self.dump_file = self.experiment_file_prefix + ".p"
        self.plot_file = self.experiment_file_prefix + ".png"
        self.results = []

    def run(self):
        start = dt.now()
        print("Starting at:", start)
        runner = self.experiment_configurer.configure()
        repeated_experiment_results = runner.run()
        if "tree" in self.experiment_config.experiment_type:
            with open(self.experiment_file_prefix + "_ACTUAL_MIXTURES_AND_BUDGETS.p", 'wb') as f:
                pickle.dump({"mixtures": repeated_experiment_results.final_mixtures_all,
                             "budgets": repeated_experiment_results.actual_costs_all,
                             "best_sols": repeated_experiment_results.best_sols_all}, f)
        elif self.experiment_config.experiment_type == "uniform":
            with open(self.experiment_file_prefix + "_PRETRAINED_UNIFORM_BASELINE.p", 'wb') as f:
                pickle.dump({"mixtures": repeated_experiment_results.final_mixtures_all,
                             "budgets": repeated_experiment_results.actual_costs_all,
                             "best_sols": repeated_experiment_results.best_sols_all}, f)
        elif self.experiment_config.experiment_type == "constant-mixture" and \
                self.experiment_config.mixture_selection_strategy == "all-individual-sources":
            with open(self.experiment_file_prefix + "_INDIVIDUAL_SRC_BASELINE.p", 'wb') as f:
                pickle.dump({"mixtures": repeated_experiment_results.final_mixtures_all,
                             "budgets": repeated_experiment_results.actual_costs_all,
                             "best_sols": repeated_experiment_results.best_sols_all}, f)
        self.results.append(repeated_experiment_results)
        end = dt.now()
        print("Ending at: %s with duration: %s" % (end, end - start))

    def dump_results(self):
        print("dumping experiment results to", self.dump_file)
        with open(self.dump_file, "wb") as f:
            pickle.dump(self, f)
