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
                        train_script_path: str):
    tested_dataset_name = DatasetManager.get_dataset_name(tested_dataset)
    run_name = ExperimentManager.get_experiment_run_name(tested_model,
                                                         image_size,
                                                         epoch_size,
                                                         batch_size)
    done_trainings = ExperimentManager.get_done_experiments(results_path,
                                                            'YOLOv6',
                                                            tested_dataset_name,
                                                            'Train')
    if run_name not in done_trainings:
        __run_training(train_script_path,
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
                                                        'YOLOv6',
                                                        tested_dataset_name,
                                                        'Test')
    if run_name not in done_tests:
        __run_testing(test_script_path,
                      results_path,
                      tested_dataset_name,
                      run_name,
                      tested_dataset,
                      image_size)


def __run_testing(yolov6_test_path: str,
                  results_path: str,
                  tested_dataset_name: str,
                  run_name: str,
                  tested_dataset: str,
                  image_size: int):
    best_model = os.path.join(results_path, 'YOLOv6', tested_dataset_name, 'Train', run_name,
                              'weights', 'best_ckpt.pt')
    test_cmd = ['python',
                yolov6_test_path,
                f'--task=val',
                f'--save_dir={results_path}/YOLOv6/{tested_dataset_name}/Test',
                f'--name={run_name}',
                f'--data={tested_dataset}',
                f'--img={image_size}',
                f'--weights={best_model}',
                ]
    subprocess.Popen(test_cmd, stdout=subprocess.PIPE).wait()


def __run_training(yolov6_train_path: str,
                   results_path: str,
                   tested_dataset_name: str,
                   run_name: str,
                   tested_dataset: str,
                   epoch_size: int,
                   batch_size: int,
                   image_size: int,
                   tested_model: str):
    train_cmd = ['python', f'{yolov6_train_path}',
                 f'--output-dir={results_path}/YOLOv6/{tested_dataset_name}/Train',
                 f'--name={run_name}',
                 f'--data-path={tested_dataset}',
                 f'--epochs={epoch_size}',
                 f'--batch-size={batch_size}',
                 f'--img-size={image_size}',
                 f'--conf-file=../external/YOLOv6/configs/{tested_model}.py']
    subprocess.Popen(train_cmd, stdout=subprocess.PIPE).wait()