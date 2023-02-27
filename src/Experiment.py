import os
import Dataset

from abc import abstractmethod



def get_experiment_run_name(tested_model: str,
                            image_size: int,
                            epoch_size: int,
                            batch_size: int) -> str:
    return f'{tested_model}_{image_size}_{epoch_size}_{batch_size}'


def get_experiment_results_path(results_path: str,
                                model_family: str,
                                tested_dataset_name: str,
                                realized_task: str):
    return os.path.join(results_path, model_family, tested_dataset_name, realized_task)


def ensure_path_exists(folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


class ExperimentManager:
    def __init__(self,
                 results_path: str,
                 model_family: str,
                 base_train_script_path: str,
                 aux_train_script_path: str,
                 test_script_path: str):

        # Fields statically the same during whole training / testing
        self.exists = True
        self.results_path = results_path
        self.model_family = model_family
        self.base_train_script_path = base_train_script_path
        self.aux_train_script_path = aux_train_script_path
        self.test_script_path = test_script_path

        # Fields needing initialization at the state
        self.done_trainings = []
        self.done_tests = []

        # Local fields changing run by run
        self.tested_dataset = 'empty_tested_dataset'
        self.run_name = 'empty_run_name'
        self.tested_dataset_name = 'empty_tested_dataset_name'

    def get_done_experiments(self,
                             realized_task: str) -> list[str]:
        results_path = get_experiment_results_path(self.results_path,
                                                   self.model_family,
                                                   self.tested_dataset_name,
                                                   realized_task)

        ensure_path_exists(results_path)
        return os.listdir(results_path)

    def conduct_experiments(self,
                            tested_dataset: str,
                            tested_model: str,
                            image_size: int,
                            epoch_size: int,
                            batch_size: int):
        self.prepare_for_experiment(tested_dataset,
                                    tested_model,
                                    image_size,
                                    epoch_size,
                                    batch_size)

        #if self.run_name not in self.done_trainings:
        self.run_training(tested_dataset,
                              epoch_size,
                              batch_size,
                              image_size,
                              tested_model)

    def conduct_testing(self,
                        tested_dataset: str,
                        tested_model: str,
                        image_size: int,
                        epoch_size: int,
                        batch_size: int):

        self.prepare_for_testing(tested_dataset,
                                 tested_model,
                                 image_size,
                                 epoch_size,
                                 batch_size)

        if self.run_name not in self.done_tests:
            self.run_testing(tested_dataset,
                             image_size)

    def prepare_for_experiment(self,
                               tested_dataset: str,
                               tested_model: str,
                               image_size: int,
                               epoch_size: int,
                               batch_size: int):
        self.tested_dataset = tested_dataset
        self.tested_dataset_name = Dataset.get_dataset_name(tested_dataset)
        self.run_name = get_experiment_run_name(tested_model,
                                                image_size,
                                                epoch_size,
                                                batch_size)
        self.done_trainings = self.get_done_experiments('Train')

    def prepare_for_testing(self,
                            tested_dataset: str,
                            tested_model: str,
                            image_size: int,
                            epoch_size: int,
                            batch_size: int):
        self.tested_dataset = tested_dataset
        self.tested_dataset_name = Dataset.get_dataset_name(tested_dataset)
        self.run_name = get_experiment_run_name(tested_model,
                                                image_size,
                                                epoch_size,
                                                batch_size)
        self.done_tests = self.get_done_experiments('Test')

    @abstractmethod
    def run_training(self,
                     tested_dataset: str,
                     epoch_size: int,
                     batch_size: int,
                     image_size: int,
                     tested_model: str):
        pass

    @abstractmethod
    def run_testing(self,
                    tested_dataset: str,
                    image_size: int):
        pass
