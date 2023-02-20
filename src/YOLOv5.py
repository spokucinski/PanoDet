import os
import subprocess
import logging

from Experiment import ExperimentManager


class YOLOv5ExperimentManager(ExperimentManager):
    def __init__(self,
                 results_path: str,
                 base_train_script_path: str,
                 aux_train_script_path: str,
                 test_script_path: str
                 ):

        logger = logging.getLogger(__name__)
        logger.info("Initialization of YOLOv5ExperimentManager started")

        try:
            if not os.path.exists(results_path):
                raise ValueError(f"Provided results_path: {results_path} does not exist!")

            if not os.path.exists(base_train_script_path):
                raise ValueError(f"Provided base_train_script_path: {base_train_script_path} does not exist!")

            if not os.path.exists(base_train_script_path):
                raise ValueError(f"Provided test_script_path: {test_script_path} does not exist!")

        except ValueError as e:
            logger.error("Error during YOLOv5ExperimentManager initialization", exc_info=True)
            logger.error("Initialization of YOLOv5ExperimentManager failed!")
            logger.error("YOLOv5 will be skipped in experiments!")
            self.exists = False
            return

        super().__init__(results_path,
                         'YOLOv5',
                         base_train_script_path,
                         aux_train_script_path,
                         test_script_path)

        logger.info("Initialization of YOLOv5ExperimentManager successful!")

    def run_testing(self,
                    tested_dataset: str,
                    image_size: int):

        try:
            best_model = os.path.join(self.results_path,
                                      'YOLOv5',
                                      self.tested_dataset_name,
                                      'Train',
                                      self.run_name,
                                      'weights',
                                      'best.pt')
            test_cmd = ['python',
                        self.test_script_path,
                        f'--task=test',
                        f'--project={self.results_path}/YOLOv5/{self.tested_dataset_name}/Test',
                        f'--name={self.run_name}',
                        f'--data={tested_dataset}',
                        f'--img={image_size}',
                        f'--weights={best_model}',
                        ]
            subprocess.Popen(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).wait()
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Exception during YOLOv5 testing! Run command: {test_cmd}")

    def run_training(self,
                     tested_dataset: str,
                     epoch_size: int,
                     batch_size: int,
                     image_size: int,
                     tested_model: str):

        try:
            train_cmd = ['python', f'{self.base_train_script_path}',
                         f'--project={self.results_path}/YOLOv5/{self.tested_dataset_name}/Train',
                         f'--name={self.run_name}',
                         f'--data={tested_dataset}',
                         f'--epochs={epoch_size}',
                         f'--batch-size={batch_size}',
                         f'--img={image_size}',
                         f'--weights={tested_model}.pt']

            subprocess.Popen(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).wait()
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Exception during YOLOv5 training! Run command: {train_cmd}")