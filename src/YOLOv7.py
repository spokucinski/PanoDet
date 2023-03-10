import os
import logging
import signal
import subprocess

from Experiment import ExperimentManager


class YOLOv7ExperimentManager(ExperimentManager):
    def __init__(self,
                 results_path: str,
                 base_train_script_path: str,
                 aux_train_script_path: str,
                 test_script_path: str):

        logger = logging.getLogger(__name__)
        logger.info("Initialization of YOLOv7ExperimentManager started")

        try:
            if not os.path.exists(results_path):
                raise ValueError(f"Provided results_path: {results_path} does not exist!")

            if not os.path.exists(base_train_script_path):
                raise ValueError(f"Provided base_train_script_path: {base_train_script_path} does not exist!")

            if not os.path.exists(aux_train_script_path):
                raise ValueError(f"Provided aux_train_script_path: {aux_train_script_path} does not exist!")

            if not os.path.exists(base_train_script_path):
                raise ValueError(f"Provided test_script_path: {test_script_path} does not exist!")

        except ValueError as e:
            logger.error("Error during YOLOv7ExperimentManager initialization", exc_info=True)
            logger.error("Initialization of YOLOv7ExperimentManager failed!")
            logger.error("YOLOv7 will be skipped in experiments!")
            self.exists = False
            return

        super().__init__(results_path,
                         'YOLOv7',
                         base_train_script_path,
                         aux_train_script_path,
                         test_script_path)

        logger.info("Initialization of YOLOv7ExperimentManager successful!")

    def run_testing(self,
                    tested_dataset: str,
                    image_size: int):

        try:
            best_model = os.path.join(self.results_path,
                                      'YOLOv7',
                                      self.tested_dataset_name,
                                      'Train',
                                      self.run_name,
                                      'weights',
                                      'best.pt')
            # CRITICAL WARNING:
            # Because of unknown reasons YOLOv7 expects the path to the model
            # to be lowercase. In "google_utils.py" while loading the model they call .lower()
            # on the file path. It has to be manually removed to be working reasonably
            # from line 21:     file = Path(str(file).strip().replace("'", '').lower())
            # to line 21:       file = Path(str(file).strip().replace("'", ''))
            test_cmd = ['python', f'{self.test_script_path}',
                        f'--data={tested_dataset}',
                        f'--img-size={image_size}',
                        f'--device=0',
                        f'--project={self.results_path}/YOLOv7/{self.tested_dataset_name}/Test',
                        f'--name={self.run_name}',
                        f'--weights={best_model}',
                        f'--task=test',
                        ]

            test_path = f'{self.results_path}/YOLOv7/{self.tested_dataset_name}/TestLog/{self.run_name}'
            if not os.path.exists(test_path):
                os.makedirs(test_path)
            test_log = open(
                f'{self.results_path}/YOLOv7/{self.tested_dataset_name}/TestLog/{self.run_name}/run_log.txt', 'a')
            test_log.write(f'TESTING LOG OF: {self.run_name}')
            test_log.flush()

            p = subprocess.Popen(test_cmd, stdout=test_log, stderr=test_log, start_new_session=True)
            p.wait(timeout=14400)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Exception during YOLOv7 testing! Run command: {test_cmd}")
            logger.error(f"Killing YOLOv7 testing!")
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            logger.error("YOLOv7 testing killed!")

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
            elif epoch_size == 200:
                previous_epoch_size = 100
                epoch_size = 100
            elif epoch_size == 300:
                previous_epoch_size = 200
                epoch_size = 100

            splitted_run_name = self.run_name.split('_')
            splitted_run_name[2] = str(previous_epoch_size)
            previous_run_name = '_'.join(splitted_run_name)

            best_previous_model_path = os.path.join(self.results_path,
                                                    'YOLOv7',
                                                    self.tested_dataset_name,
                                                    'Train',
                                                    previous_run_name,
                                                    'weights',
                                                    'best.pt')

            old_path_exists = os.path.exists(best_previous_model_path)

            if "6" in tested_model:

                if old_path_exists:
                    train_cmd = ['python', f'{self.aux_train_script_path}',
                                 f'--device=0',
                                 f'--weights={best_previous_model_path}',
                                 f'--project={self.results_path}/YOLOv7/{self.tested_dataset_name}/Train',
                                 f'--name={self.run_name}',
                                 f'--data={tested_dataset}',
                                 f'--epochs={epoch_size}',
                                 f'--batch-size={batch_size}',
                                 f'--img-size={image_size}',
                                 f'--hyp=../external/YOLOv7/data/hyp.scratch.p6.yaml',
                                 f'--cfg=../external/YOLOv7/cfg/training/{tested_model}.yaml']
                else:

                    if previous_epoch_size == 0:
                        epoch_size = 50
                    elif previous_epoch_size == 50:
                        epoch_size = 100
                    elif previous_epoch_size == 100:
                        epoch_size = 200
                    elif previous_epoch_size == 200:
                        epoch_size = 300
                    elif previous_epoch_size == 300:
                        epoch_size = 500

                    train_cmd = ['python', f'{self.aux_train_script_path}',
                                 f'--device=0',
                                 f'--weights=""',
                                 f'--project={self.results_path}/YOLOv7/{self.tested_dataset_name}/Train',
                                 f'--name={self.run_name}',
                                 f'--data={tested_dataset}',
                                 f'--epochs={epoch_size}',
                                 f'--batch-size={batch_size}',
                                 f'--img-size={image_size}',
                                 f'--hyp=../external/YOLOv7/data/hyp.scratch.p6.yaml',
                                 f'--cfg=../external/YOLOv7/cfg/training/{tested_model}.yaml']
            else:
                if old_path_exists:
                    train_cmd = ['python', f'{self.base_train_script_path}',
                                 f'--device=0',
                                 f'--weights={best_previous_model_path}',
                                 f'--project={self.results_path}/YOLOv7/{self.tested_dataset_name}/Train',
                                 f'--name={self.run_name}',
                                 f'--data={tested_dataset}',
                                 f'--epochs={epoch_size}',
                                 f'--batch-size={batch_size}',
                                 f'--img-size={image_size}',
                                 f'--hyp=../external/YOLOv7/data/hyp.scratch.p5.yaml',
                                 f'--cfg=../external/YOLOv7/cfg/training/{tested_model}.yaml']
                else:

                    if previous_epoch_size == 0:
                        epoch_size = 50
                    elif previous_epoch_size == 50:
                        epoch_size = 100
                    elif previous_epoch_size == 100:
                        epoch_size = 200
                    elif previous_epoch_size == 200:
                        epoch_size = 300
                    elif previous_epoch_size == 300:
                        epoch_size = 500

                    train_cmd = ['python', f'{self.base_train_script_path}',
                                 f'--device=0',
                                 f'--weights=""',
                                 f'--project={self.results_path}/YOLOv7/{self.tested_dataset_name}/Train',
                                 f'--name={self.run_name}',
                                 f'--data={tested_dataset}',
                                 f'--epochs={epoch_size}',
                                 f'--batch-size={batch_size}',
                                 f'--img-size={image_size}',
                                 f'--hyp=../external/YOLOv7/data/hyp.scratch.p5.yaml',
                                 f'--cfg=../external/YOLOv7/cfg/training/{tested_model}.yaml']

            train_path = f'{self.results_path}/YOLOv7/{self.tested_dataset_name}/TrainLog/{self.run_name}'
            if not os.path.exists(train_path):
                os.makedirs(train_path)
            train_log = open(
                f'{self.results_path}/YOLOv7/{self.tested_dataset_name}/TrainLog/{self.run_name}/run_log.txt', 'a')
            train_log.write(f'TRAINING LOG OF: {self.run_name}')
            train_log.flush()

            p = subprocess.Popen(train_cmd, stdout=train_log, stderr=train_log, start_new_session=True)
            p.wait(timeout=14400)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Exception during YOLOv7 training! Run command: {train_cmd}")
            logger.error(f"Killing YOLOv7 training!")
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            logger.error("YOLOv7 training killed!")
