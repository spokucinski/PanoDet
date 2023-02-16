import subprocess
from Experiment import ExperimentManager
#from Dataset import Data
import os


class YOLOv5ExperimentManager(ExperimentManager):
    def __init__(self,
                 results_path: str,
                 base_train_script_path: str,
                 aux_train_script_path: str,
                 test_script_path: str
                 ):
        super().__init__(results_path,
                         'YOLOv5',
                         base_train_script_path,
                         aux_train_script_path,
                         test_script_path)

    def run_testing(self,
                    tested_dataset: str,
                    image_size: int):

        best_model = os.path.join(self.results_path, 'YOLOv5', self.tested_dataset_name, 'Train', self.run_name, 'weights', 'best.pt')
        test_cmd = ['python',
                    self.test_script_path,
                    f'--task=test',
                    f'--project={self.results_path}/YOLOv5/{self.tested_dataset_name}/Test',
                    f'--name={self.run_name}',
                    f'--data={tested_dataset}',
                    f'--img={image_size}',
                    f'--weights={best_model}',
                    ]
        subprocess.Popen(test_cmd, stdout=subprocess.PIPE).wait()

    def run_training(self,
                     tested_dataset: str,
                     epoch_size: int,
                     batch_size: int,
                     image_size: int,
                     tested_model: str):

        train_cmd = ['python', f'{self.base_train_script_path}',
                     f'--project={self.results_path}/YOLOv5/{self.tested_dataset_name}/Train',
                     f'--name={self.run_name}',
                     f'--data={tested_dataset}',
                     f'--epochs={epoch_size}',
                     f'--batch-size={batch_size}',
                     f'--img={image_size}',
                     f'--weights={tested_model}.pt']

        subprocess.Popen(train_cmd, stdout=subprocess.PIPE).wait()

# def conduct_experiments(tested_dataset: str,
#                         tested_model: str,
#                         image_size: int,
#                         epoch_size: int,
#                         batch_size: int,
#                         results_path: str,
#                         train_script_path: str):
#     tested_dataset_name = DatasetManager.get_dataset_name(tested_dataset)
#     run_name = ExperimentManager.get_experiment_run_name(tested_model,
#                                                          image_size,
#                                                          epoch_size,
#                                                          batch_size)
#     done_trainings = ExperimentManager.get_done_experiments(results_path,
#                                                             'YOLOv5',
#                                                             tested_dataset_name,
#                                                             'Train')
#     if run_name not in done_trainings:
#         __run_training(train_script_path,
#                        results_path,
#                        tested_dataset_name,
#                        run_name,
#                        tested_dataset,
#                        epoch_size,
#                        batch_size,
#                        image_size,
#                        tested_model)
#
#
# def conduct_tests(tested_dataset: str,
#                   tested_model: str,
#                   image_size: int,
#                   epoch_size: int,
#                   batch_size: int,
#                   results_path: str,
#                   test_script_path: str):
#     tested_dataset_name = DatasetManager.get_dataset_name(tested_dataset)
#     run_name = ExperimentManager.get_experiment_run_name(tested_model,
#                                                          image_size,
#                                                          epoch_size,
#                                                          batch_size)
#     done_tests = ExperimentManager.get_done_experiments(results_path,
#                                                         'YOLOv5',
#                                                         tested_dataset_name,
#                                                         'Test')
#     if run_name not in done_tests:
#         __run_testing(test_script_path,
#                       results_path,
#                       tested_dataset_name,
#                       run_name,
#                       tested_dataset,
#                       image_size)
