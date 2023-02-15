import os
import subprocess
import DatasetManager
import ExperimentManager
import YOLOv5

# Initial notes
# Main development tool is PyCharm
# Project should be configured with the "src" folder marked as the Source Root Folder
# Any paths are relative to this source root
# Datasets are expected to be provided with a yaml file describing the datasets content and its paths to train/val/test subsets

# Manual configuration part
project_name = 'SimPanoDet'
dataset_file_extension = '.yaml'
results_path = '../results'
epochs = [1]  # [50, 100, 150, 200, 300]
image_sizes = [1024]  # [1024, 512, 256]
batch_sizes = [1]  # [96, 64, 32, -1]

v5_datasets_path = '../datasets/YOLOv5'
yolov5_train_path = '/home/sebex/Repos/PanoDet/external/YOLOv5/train.py'
yolov5_val_path = '/home/sebex/Repos/PanoDet/external/YOLOv5/val.py'
tested_yolov5_models = ['yolov5n', 'yolov5s']  # ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']

v6_datasets_path = '../datasets/YOLOv6'
yolov6_train_path = '/home/sebex/Repos/PanoDet/external/YOLOv6/tools/train.py'
yolov6_val_path = '/home/sebex/Repos/PanoDet/external/YOLOv6/tools/eval.py'
tested_yolov6_models = ['yolov6n',
                        'yolov6n6']  # ['yolov6n', 'yolov6n6', 'yolov6s', 'yolov6s6', 'yolov6m', 'yolov6m6', 'yolov6l', 'yolov6l6']

v7_datasets_path = '../datasets/YOLOv5'  # Dataset format for v7 is the same as for v5
yolov7_train_path = '/home/sebex/Repos/PanoDet/external/YOLOv7/train.py'
yolov7_val_path = '/home/sebex/Repos/PanoDet/external/YOLOv5/val.py'
tested_yolov7_models = ['yolov7', 'yolov7x', 'yolov7-w6', 'yolov7-e6', 'yolov7-d6', 'yolov7-e6e']

yolov8_train_path = ''
yolov8_val_path = ''
tested_yolov8_models = ['yolov5n', 'yolov5s']

# Testing on selected model
# tested_models = ['/home/sebex/Repos/yolov5-SimPanoDet/Experiments/SimPanoDet/Datasets/WiMR_BB_70-20-10/WiMR_BB_70-20-10.yaml-yolov5x-1024-200--1/weights/best']
# for tested_model in tested_models:
#     cmd = ['python', '/home/sebex/Repos/yolov5-SimPanoDet/detect.py', f'--weights={tested_model}.pt', f'--source=/home/sebex/Repos/yolov5-SimPanoDet/Experiments/Showcase/Input',
#            f'--img=1024']
#     subprocess.Popen(cmd, stdout=subprocess.PIPE).wait()


# YOLOv5 Experiments
v5_datasets_to_test = DatasetManager.get_dataset_file_paths(v5_datasets_path, dataset_file_extension)

for tested_dataset in v5_datasets_to_test:
    for tested_model in tested_yolov5_models:
        for image_size in image_sizes:
            for epoch_size in epochs:
                for batch_size in batch_sizes:
                    YOLOv5.conduct_experiments(tested_dataset,
                                               tested_model,
                                               image_size,
                                               epoch_size,
                                               batch_size,
                                               results_path,
                                               yolov5_train_path)

                    YOLOv5.conduct_tests(tested_dataset,
                                         tested_model,
                                         image_size,
                                         epoch_size,
                                         batch_size,
                                         results_path,
                                         yolov5_val_path)

# YOLOv6 Experiments
v6_datasets_to_test = []
for root, dirs, files in os.walk(v6_datasets_path):
    for file in files:
        if file.endswith(".yaml"):
            v6_datasets_to_test.append(os.path.join(root, file))

for tested_dataset in v6_datasets_to_test:
    # YOLOv6 Experiments
    for tested_model in tested_yolov6_models:
        for image_size in image_sizes:
            for epoch_size in epochs:
                for batch_size in batch_sizes:

                    # YOLOv6 does not support auto batch size
                    if batch_size == -1:
                        continue

                    # Split the path to datset with use of os separator, take the last separated element ('datasetname.yaml')
                    # split it again with the dot and take the first part - name of the .yaml file ('datasetname')
                    tested_dataset_name = tested_dataset.split(os.sep)[-1].split('.')[0]
                    run_name = f'{tested_model}_{image_size}_{epoch_size}_{batch_size}'

                    if not os.path.exists(os.path.join(results_path, 'YOLOv6', tested_dataset_name, 'Train')):
                        os.makedirs(os.path.join(results_path, 'YOLOv6', tested_dataset_name, 'Train'))
                    done_trainings = os.listdir(os.path.join(results_path, 'YOLOv6', tested_dataset_name, 'Train'))
                    if run_name not in done_trainings:
                        train_cmd = ['python', f'{yolov6_train_path}',
                                     f'--output-dir={results_path}/YOLOv6/{tested_dataset_name}/Train',
                                     f'--name={run_name}',
                                     f'--data-path={tested_dataset}',
                                     f'--epochs={epoch_size}',
                                     f'--batch-size={batch_size}',
                                     f'--img-size={image_size}',
                                     f'--conf-file=../external/YOLOv6/configs/{tested_model}.py']
                        subprocess.Popen(train_cmd, stdout=subprocess.PIPE).wait()

                    if not os.path.exists(os.path.join(results_path, 'YOLOv6', tested_dataset_name, 'Test')):
                        os.makedirs(os.path.join(results_path, 'YOLOv6', tested_dataset_name, 'Test'))
                    done_tests = os.listdir(os.path.join(results_path, 'YOLOv6', tested_dataset_name, 'Test'))
                    if run_name not in done_tests:
                        best_model = os.path.join(results_path, 'YOLOv6', tested_dataset_name, 'Train', run_name,
                                                  'weights', 'best_ckpt.pt')
                        test_cmd = ['python',
                                    yolov6_val_path,
                                    f'--task=val',
                                    f'--save_dir={results_path}/YOLOv6/{tested_dataset_name}/Test',
                                    f'--name={run_name}',
                                    f'--data={tested_dataset}',
                                    f'--img={image_size}',
                                    f'--weights={best_model}',
                                    # '--save-txt'
                                    ]
                        subprocess.Popen(test_cmd, stdout=subprocess.PIPE).wait()

# YOLOv7 Experiments
v7_datasets_to_test = []
for root, dirs, files in os.walk(v7_datasets_path):
    for file in files:
        if file.endswith(".yaml"):
            v7_datasets_to_test.append(os.path.join(root, file))

for tested_dataset in v7_datasets_to_test:
    # YOLOv6 Experiments
    for tested_model in tested_yolov7_models:
        for image_size in image_sizes:
            for epoch_size in epochs:
                for batch_size in batch_sizes:

                    # Split the path to datset with use of os separator, take the last separated element ('datasetname.yaml')
                    # split it again with the dot and take the first part - name of the .yaml file ('datasetname')
                    tested_dataset_name = tested_dataset.split(os.sep)[-1].split('.')[0]
                    run_name = f'{tested_model}_{image_size}_{epoch_size}_{batch_size}'

                    if not os.path.exists(os.path.join(results_path, 'YOLOv7', tested_dataset_name, 'Train')):
                        os.makedirs(os.path.join(results_path, 'YOLOv7', tested_dataset_name, 'Train'))
                    done_trainings = os.listdir(os.path.join(results_path, 'YOLOv7', tested_dataset_name, 'Train'))
                    if run_name not in done_trainings:
                        train_cmd = ['python', f'{yolov7_train_path}',
                                     f'--output-dir={results_path}/YOLOv6/{tested_dataset_name}/Train',
                                     f'--name={run_name}',
                                     f'--data-path={tested_dataset}',
                                     f'--epochs={epoch_size}',
                                     f'--batch-size={batch_size}',
                                     f'--img-size={image_size}',
                                     f'--conf-file=../external/YOLOv6/configs/{tested_model}.py']
                        subprocess.Popen(train_cmd, stdout=subprocess.PIPE).wait()

                    if not os.path.exists(os.path.join(results_path, 'YOLOv6', tested_dataset_name, 'Test')):
                        os.makedirs(os.path.join(results_path, 'YOLOv6', tested_dataset_name, 'Test'))
                    done_tests = os.listdir(os.path.join(results_path, 'YOLOv6', tested_dataset_name, 'Test'))
                    if run_name not in done_tests:
                        best_model = os.path.join(results_path, 'YOLOv6', tested_dataset_name, 'Train', run_name,
                                                  'weights', 'best_ckpt.pt')
                        test_cmd = ['python',
                                    yolov6_val_path,
                                    f'--task=val',
                                    f'--save_dir={results_path}/YOLOv6/{tested_dataset_name}/Test',
                                    f'--name={run_name}',
                                    f'--data={tested_dataset}',
                                    f'--img={image_size}',
                                    f'--weights={best_model}',
                                    # '--save-txt'
                                    ]
                        subprocess.Popen(test_cmd, stdout=subprocess.PIPE).wait()

# Uncomment if you want to shut down the computer after experiments
# os.system('shutdown now -h')
