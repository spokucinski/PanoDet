import os
import subprocess
import logging
import signal

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
                        f'--device=cpu', # Testing intentionally run on CPU - as it was done on a terminal device without GPU
                        f'--project={self.results_path}/YOLOv5/{self.tested_dataset_name}/Test',
                        f'--name={self.run_name}',
                        f'--data={tested_dataset}',
                        f'--img={image_size}',
                        f'--weights={best_model}']

            test_path = f'{self.results_path}/YOLOv5/{self.tested_dataset_name}/TestLog/{self.run_name}'
            if not os.path.exists(test_path):
                os.makedirs(test_path)
            test_log = open(f'{self.results_path}/YOLOv5/{self.tested_dataset_name}/TestLog/{self.run_name}/run_log.txt', 'a')
            test_log.write(f'TESTING LOG OF: {self.run_name}')
            test_log.flush()
            p = subprocess.Popen(test_cmd, stdout=test_log, stderr=test_log, start_new_session=True)
            p.wait(timeout=14400)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Exception during YOLOv5 testing! Run command: {test_cmd}")
            logger.error(f"Killing YOLOv5 testing!")
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            logger.error("YOLOv5 testing killed!")

    def run_training(self,
                     tested_dataset: str,
                     epoch_size: int,
                     batch_size: int,
                     image_size: int,
                     tested_model: str):

        try:

            previous_epoch_size = 0
            if epoch_size == 100:
                previous_epoch_size = 50
                epoch_size = 50
            if epoch_size == 200:
                previous_epoch_size = 100
                epoch_size = 100
            if epoch_size == 300:
                previous_epoch_size = 200
                epoch_size = 100

            splitted_run_name = self.run_name.split('_')
            splitted_run_name[2] = str(previous_epoch_size)
            previous_run_name = '_'.join(splitted_run_name)

            best_previous_model_path = os.path.join(self.results_path,
                                                    'YOLOv5',
                                                    self.tested_dataset_name,
                                                    'Train',
                                                    previous_run_name,
                                                    'weights',
                                                    'best.pt')

            old_path_exists = os.path.exists(best_previous_model_path)

            if old_path_exists:
                train_cmd = ['python', f'{self.base_train_script_path}',
                         f'--project={self.results_path}/YOLOv5/{self.tested_dataset_name}/Train',
                         f'--name={self.run_name}',
                         f'--data={tested_dataset}',
                         f'--epochs={epoch_size}',
                         f'--batch-size={batch_size}',
                         f'--img={image_size}',
                         f'--weights={best_previous_model_path}']
            else:
                train_cmd = ['python', f'{self.base_train_script_path}',
                         f'--project={self.results_path}/YOLOv5/{self.tested_dataset_name}/Train',
                         f'--name={self.run_name}',
                         f'--data={tested_dataset}',
                         f'--epochs={epoch_size}',
                         f'--batch-size={batch_size}',
                         f'--img={image_size}',
                         f'--weights={tested_model}.pt']

            train_path = f'{self.results_path}/YOLOv5/{self.tested_dataset_name}/TrainLog/{self.run_name}'
            if not os.path.exists(train_path):
                os.makedirs(train_path)
            train_log = open(
                f'{self.results_path}/YOLOv5/{self.tested_dataset_name}/TrainLog/{self.run_name}/run_log.txt', 'a')
            train_log.write(f'TRAINING LOG OF: {self.run_name}')
            train_log.flush()
            p = subprocess.Popen(train_cmd, stdout=train_log, stderr=train_log, start_new_session=True)
            p.wait(timeout=14400)

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Exception during YOLOv5 training! Run command: {train_cmd}")
            logger.error(f"Killing YOLOv5 training!")
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            logger.error("YOLOv5 training killed!")
