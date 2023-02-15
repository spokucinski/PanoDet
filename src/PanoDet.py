import os
import subprocess
import DatasetManager
import ExperimentManager

import YOLOv5
import YOLOv6
import YOLOv7
import YOLOv8



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
yolov5_train_path = '../external/YOLOv5/train.py'
yolov5_val_path = '../external/YOLOv5/val.py'
tested_yolov5_models = ['yolov5n', 'yolov5s'] #['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']

v6_datasets_path = '../datasets/YOLOv6'
yolov6_train_path = '../external/YOLOv6/tools/train.py'
yolov6_val_path = '../external/YOLOv6/tools/eval.py'
tested_yolov6_models = ['yolov6n', 'yolov6n6']#['yolov6n', 'yolov6n6', 'yolov6s', 'yolov6s6', 'yolov6m', 'yolov6m6', 'yolov6l', 'yolov6l6']

v7_datasets_path = '../datasets/YOLOv7'
yolov7_train_path = '../external/YOLOv7/train.py'
yolov7_train_aux_path = '../external/YOLOv7/train_aux.py'
yolov7_val_path = '../external/YOLOv7/test.py'
tested_yolov7_models = ['yolov7'] #['yolov7', 'yolov7x', 'yolov7-w6', 'yolov7-e6', 'yolov7-d6', 'yolov7-e6e']

v8_datasets_path = '../datasets/YOLOv8'
tested_yolov8_models = ['yolov8n'] #['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']

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
v6_datasets_to_test = DatasetManager.get_dataset_file_paths(v6_datasets_path, dataset_file_extension)
for tested_dataset in v6_datasets_to_test:
    for tested_model in tested_yolov6_models:
        for image_size in image_sizes:
            for epoch_size in epochs:
                for batch_size in batch_sizes:

                    # YOLOv6 does not support auto batch size
                    if batch_size == -1:
                        continue

                    YOLOv6.conduct_experiments(tested_dataset,
                                               tested_model,
                                               image_size,
                                               epoch_size,
                                               batch_size,
                                               results_path,
                                               yolov6_train_path)

                    YOLOv6.conduct_tests(tested_dataset,
                                         tested_model,
                                         image_size,
                                         epoch_size,
                                         batch_size,
                                         results_path,
                                         yolov6_val_path)


# YOLOv7 Experiments
v7_datasets_to_test = DatasetManager.get_dataset_file_paths(v7_datasets_path, dataset_file_extension)
for tested_dataset in v7_datasets_to_test:
    for tested_model in tested_yolov7_models:
        for image_size in image_sizes:
            for epoch_size in epochs:
                for batch_size in batch_sizes:

                    # YOLOv7 does not support auto batch size
                    if batch_size == -1:
                        continue

                    YOLOv7.conduct_experiments(tested_dataset,
                                               tested_model,
                                               image_size,
                                               epoch_size,
                                               batch_size,
                                               results_path,
                                               yolov7_train_path,
                                               yolov7_train_aux_path)

                    YOLOv7.conduct_tests(tested_dataset,
                                         tested_model,
                                         image_size,
                                         epoch_size,
                                         batch_size,
                                         results_path,
                                         yolov7_val_path)


# YOLOv8 Experiments
v8_datasets_to_test = DatasetManager.get_dataset_file_paths(v8_datasets_path, dataset_file_extension)
for tested_dataset in v8_datasets_to_test:
    for tested_model in tested_yolov8_models:
        for image_size in image_sizes:
            for epoch_size in epochs:
                for batch_size in batch_sizes:

                    YOLOv8.conduct_experiments(tested_dataset,
                                               tested_model,
                                               image_size,
                                               epoch_size,
                                               batch_size,
                                               results_path)

                    YOLOv8.conduct_tests(tested_dataset,
                                         tested_model,
                                         image_size,
                                         epoch_size,
                                         batch_size,
                                         results_path)


# Uncomment if you want to shut down the computer after experiments
# os.system('shutdown now -h')
