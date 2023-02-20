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
