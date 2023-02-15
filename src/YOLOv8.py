from ultralytics import YOLO
import ExperimentManager
import DatasetManager
import os


def conduct_experiments(tested_dataset: str,
                        tested_model: str,
                        image_size: int,
                        epoch_size: int,
                        batch_size: int,
                        results_path: str):
    tested_dataset_name = DatasetManager.get_dataset_name(tested_dataset)
    run_name = ExperimentManager.get_experiment_run_name(tested_model,
                                                         image_size,
                                                         epoch_size,
                                                         batch_size)
    done_trainings = ExperimentManager.get_done_experiments(results_path,
                                                            'YOLOv8',
                                                            tested_dataset_name,
                                                            'Train')
    if run_name not in done_trainings:
        __run_training(results_path,
                       tested_dataset_name,
                       run_name,
                       tested_dataset,
                       epoch_size,
                       batch_size,
                       image_size,
                       tested_model)


def conduct_tests(tested_dataset: str,
                  tested_model: str,
                  image_size: int,
                  epoch_size: int,
                  batch_size: int,
                  results_path: str):
    tested_dataset_name = DatasetManager.get_dataset_name(tested_dataset)
    run_name = ExperimentManager.get_experiment_run_name(tested_model,
                                                         image_size,
                                                         epoch_size,
                                                         batch_size)
    done_tests = ExperimentManager.get_done_experiments(results_path,
                                                        'YOLOv8',
                                                        tested_dataset_name,
                                                        'Test')
    if run_name not in done_tests:
        __run_testing(results_path,
                      tested_dataset_name,
                      run_name,
                      tested_dataset,
                      image_size)


def __run_testing(results_path: str,
                  tested_dataset_name: str,
                  run_name: str,
                  tested_dataset: str,
                  image_size: int):
    best_model = os.path.join(results_path, 'YOLOv8', tested_dataset_name, 'Train', run_name, 'weights', 'best.pt')
    model = YOLO(model=best_model)
    model.val(task='val',
              project=f'{results_path}/YOLOv8/{tested_dataset_name}/Test',
              name=run_name,
              data=tested_dataset,
              imgsz=image_size)


def __run_training(results_path: str,
                   tested_dataset_name: str,
                   run_name: str,
                   tested_dataset: str,
                   epoch_size: int,
                   batch_size: int,
                   tested_model: str):
    model = YOLO(model=f'{tested_model}.yaml')
    model.train(project=f'{results_path}/YOLOv8/{tested_dataset_name}/Train',
                name=run_name,
                data=tested_dataset,
                epochs=epoch_size,
                batch=batch_size)
