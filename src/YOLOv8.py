import os
import logging

from ultralytics import YOLO
from Experiment import ExperimentManager


class YOLOv8ExperimentManager(ExperimentManager):
    def __init__(self,
                 results_path: str,
                 base_train_script_path: str,
                 aux_train_script_path: str,
                 test_script_path: str):

        logger = logging.getLogger(__name__)
        logger.info("Initialization of YOLOv7ExperimentManager started")

        super().__init__(results_path,
                         'YOLOv8',
                         base_train_script_path,
                         aux_train_script_path,
                         test_script_path)

        logger.info("Initialization of YOLOv8ExperimentManager successful!")

    def run_testing(self,
                    tested_dataset: str,
                    image_size: int):

        try:
            best_model = os.path.join(self.results_path,
                                      'YOLOv8',
                                      self.tested_dataset_name,
                                      'Train',
                                      self.run_name,
                                      'weights',
                                      'best.pt')
            model = YOLO(model=best_model)
            model.val(task='val',
                      project=f'{self.results_path}/YOLOv8/{self.tested_dataset_name}/Test',
                      name=self.run_name,
                      data=tested_dataset,
                      imgsz=image_size)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Exception during YOLOv8 testing! "
                         f"Model: {best_model} "
                         f"Task:{'val'} "
                         f"Project:{self.results_path}/YOLOv8/{self.tested_dataset_name}/Test "
                         f"Name:{self.run_name} "
                         f"Data: {tested_dataset} "
                         f"Imgsz: {image_size}")
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
                                                    'YOLOv8',
                                                    self.tested_dataset_name,
                                                    'Train',
                                                    previous_run_name,
                                                    'weights',
                                                    'best.pt')

            old_path_exists = os.path.exists(best_previous_model_path)

            if old_path_exists:
                model = YOLO(model=f'{best_previous_model_path}')
                model.train(project=f'{self.results_path}/YOLOv8/{self.tested_dataset_name}/Train',
                            name=self.run_name,
                            data=tested_dataset,
                            epochs=epoch_size,
                            batch=batch_size)
            else:
                model = YOLO(model=f'{tested_model}.yaml')
                model.train(project=f'{self.results_path}/YOLOv8/{self.tested_dataset_name}/Train',
                            name=self.run_name,
                            data=tested_dataset,
                            epochs=epoch_size,
                            batch=batch_size)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Exception during YOLOv8 training! "
                         f"Model: {tested_model}.yaml "
                         f"Project: {self.results_path}/YOLOv8/{self.tested_dataset_name}/Train "
                         f"Name: {self.run_name} "
                         f"Data: {tested_dataset} "
                         f"Epochs: {epoch_size} "
                         f"Batch: {batch_size}")
