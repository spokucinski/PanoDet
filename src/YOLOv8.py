from ultralytics import YOLO
from Experiment import ExperimentManager
import os


class YOLOv8ExperimentManager(ExperimentManager):
    def __init__(self,
                 results_path: str,
                 base_train_script_path: str,
                 aux_train_script_path: str,
                 test_script_path: str):
        super().__init__(results_path,
                         'YOLOv8',
                         base_train_script_path,
                         aux_train_script_path,
                         test_script_path)

    def run_testing(self,
                    tested_dataset: str,
                    image_size: int):
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

    def run_training(self,
                     tested_dataset: str,
                     epoch_size: int,
                     batch_size: int,
                     image_size: int,
                     tested_model: str):
        model = YOLO(model=f'{tested_model}.yaml')
        model.train(project=f'{self.results_path}/YOLOv8/{self.tested_dataset_name}/Train',
                    name=self.run_name,
                    data=tested_dataset,
                    epochs=epoch_size,
                    batch=batch_size)
