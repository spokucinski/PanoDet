import subprocess
import Experiment
import Dataset
import os


def conduct_experiments(tested_dataset: str,
                        tested_model: str,
                        image_size: int,
                        epoch_size: int,
                        batch_size: int,
                        results_path: str,
                        train_script_path: str,
                        aux_train_script_path: str,):
    tested_dataset_name = DatasetManager.get_dataset_name(tested_dataset)
    run_name = ExperimentManager.get_experiment_run_name(tested_model,
                                                         image_size,
                                                         epoch_size,
                                                         batch_size)
    done_trainings = ExperimentManager.get_done_experiments(results_path,
                                                            'YOLOv7',
                                                            tested_dataset_name,
                                                            'Train')
    if run_name not in done_trainings:
        __run_training(train_script_path,
                       aux_train_script_path,
                       results_path,
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
                  results_path: str,
                  test_script_path: str):
    tested_dataset_name = DatasetManager.get_dataset_name(tested_dataset)
    run_name = ExperimentManager.get_experiment_run_name(tested_model,
                                                         image_size,
                                                         epoch_size,
                                                         batch_size)
    done_tests = ExperimentManager.get_done_experiments(results_path,
                                                        'YOLOv7',
                                                        tested_dataset_name,
                                                        'Test')
    if run_name not in done_tests:
        __run_testing(test_script_path,
                      results_path,
                      tested_dataset_name,
                      run_name,
                      tested_dataset,
                      image_size)


def __run_testing(yolov7_val_path: str,
                  results_path: str,
                  tested_dataset_name: str,
                  run_name: str,
                  tested_dataset: str,
                  image_size: int):
    best_model = os.path.join(results_path, 'YOLOv7', tested_dataset_name, 'Train', run_name,
                              'weights', 'best.pt')
    # CRITICAL WARNING:
    # Because of unknown reasons YOLOv7 expects the path to the model
    # to be lowercase. In "google_utils.py" while loading the model they call .lower()
    # on the file path. It has to be manually removed to be working reasonably
    # from line 21:     file = Path(str(file).strip().replace("'", '').lower())
    # to line 21:       file = Path(str(file).strip().replace("'", ''))
    test_cmd = ['python', f'{yolov7_val_path}',
                f'--data={tested_dataset}',
                f'--img-size={image_size}',
                f'--device=0',
                f'--project={results_path}/YOLOv7/{tested_dataset_name}/Test',
                f'--name={run_name}',
                f'--weights={best_model}',
                f'--task=test',
                ]
    subprocess.Popen(test_cmd, stdout=subprocess.PIPE).wait()


def __run_training(yolov7_train_path: str,
                   yolov7_train_aux_path: str,
                   results_path: str,
                   tested_dataset_name: str,
                   run_name: str,
                   tested_dataset: str,
                   epoch_size: int,
                   batch_size: int,
                   image_size: int,
                   tested_model: str):
    if "6" in tested_model:
        train_cmd = ['python', f'{yolov7_train_aux_path}',
                     f'--device=0',
                     f'--weights=""',
                     f'--project={results_path}/YOLOv7/{tested_dataset_name}/Train',
                     f'--name={run_name}',
                     f'--data={tested_dataset}',
                     f'--epochs={epoch_size}',
                     f'--batch-size={batch_size}',
                     f'--img-size="{image_size}',
                     f'--hyp=../external/YOLOv7/data/hyp.scratch.p6.yaml',
                     f'--cfg=../external/YOLOv7/cfg/training/{tested_model}.yaml']
    else:
        train_cmd = ['python', f'{yolov7_train_path}',
                     f'--device=0',
                     f'--weights=""',
                     f'--project={results_path}/YOLOv7/{tested_dataset_name}/Train',
                     f'--name={run_name}',
                     f'--data={tested_dataset}',
                     f'--epochs={epoch_size}',
                     f'--batch-size={batch_size}',
                     f'--img-size={image_size}',
                     f'--hyp=../external/YOLOv7/data/hyp.scratch.p5.yaml',
                     f'--cfg=../external/YOLOv7/cfg/training/{tested_model}.yaml']
    subprocess.Popen(train_cmd, stdout=subprocess.PIPE).wait()