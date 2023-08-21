import Dataset
import logging
import os

from YOLOv5 import YOLOv5ExperimentManager
from YOLOv6 import YOLOv6ExperimentManager
from YOLOv7 import YOLOv7ExperimentManager
from YOLOv8 import YOLOv8ExperimentManager

# Initial notes
# Main development tool is PyCharm
# Project should be configured with the "src" folder marked as the Source Root Folder
# Any paths are relative to this source root
# Datasets are expected to be provided with a yaml file
# describing the datasets content and its paths to train/val/test subsets

logging.basicConfig(level=logging.INFO, filename='../results/log.log', filemode='a',
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)
logger.info("Starting experiments!")
logger.info("Reading start parameters...")

logger.info("Requested experiment parameters:")
# Manual configuration part
project_name = 'PanoDet'
logger.info(f'Project name: {project_name}')

dataset_file_extension = '.yaml'
logger.info(f'Dataset file extension: {dataset_file_extension}')

results_path = '../results'
logger.info(f'Results saved to: {results_path}')

datasets_path = '../datasets'
logger.info(f'Datasets will be searched for in: {datasets_path}')


epochs = [50] #[100, 200, 300]
logger.info(f'Epochs: {epochs}')

image_sizes = [512, 1024] #[256, 512, 1024]
logger.info(f'Image sizes: {image_sizes}')

batch_sizes = [1, 2, 4, 8]
logger.info(f'Batch sizes: {batch_sizes}')

tested_yolov5_models = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
logger.info(f'YOLOv5 models: {tested_yolov5_models}')

tested_yolov6_models = ['yolov6n', 'yolov6n6', 'yolov6s', 'yolov6s6', 'yolov6m', 'yolov6m6', 'yolov6l', 'yolov6l6']
logger.info(f'YOLOv6 models: {tested_yolov6_models}')

tested_yolov7_models = ['yolov7', 'yolov7x', 'yolov7-w6', 'yolov7-e6', 'yolov7-d6', 'yolov7-e6e']
logger.info(f'YOLOv7 models: {tested_yolov7_models}')

tested_yolov8_models = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
logger.info(f'YOLOv8 models: {tested_yolov8_models}')

logger.info("Parameters reading complete!")
logger.info("Initializing experiment managers started...")

yolov5 = YOLOv5ExperimentManager(results_path,
                                 '../external/YOLOv5/train.py',
                                 None,
                                 '../external/YOLOv5/val.py')

yolov6 = YOLOv6ExperimentManager(results_path,
                                 '../external/YOLOv6/tools/train.py',
                                 None,
                                 '../external/YOLOv6/tools/eval.py')

yolov7 = YOLOv7ExperimentManager(results_path,
                                 '../external/YOLOv7/train.py',
                                 '../external/YOLOv7/train_aux.py',
                                 '../external/YOLOv7/test.py')

yolov8 = YOLOv8ExperimentManager(results_path,
                                 None,
                                 None,
                                 None)

logger.info("Experiment managers initialization ended!")

v5_datasets_to_test = Dataset.get_dataset_file_paths(f"{datasets_path}/YOLOv5", dataset_file_extension)
logger.info(f"{len(v5_datasets_to_test)} YOLOv5 datasets found: {v5_datasets_to_test}")

v6_datasets_to_test = Dataset.get_dataset_file_paths(f"{datasets_path}/YOLOv6", dataset_file_extension)
logger.info(f"{len(v6_datasets_to_test)} YOLOv6 datasets found: {v6_datasets_to_test}")

v7_datasets_to_test = Dataset.get_dataset_file_paths(f"{datasets_path}/YOLOv7", dataset_file_extension)
logger.info(f"{len(v7_datasets_to_test)} YOLOv7 datasets found: {v7_datasets_to_test}")

v8_datasets_to_test = Dataset.get_dataset_file_paths(f"{datasets_path}/YOLOv8", dataset_file_extension)
logger.info(f"{len(v8_datasets_to_test)} YOLOv8 datasets found: {v8_datasets_to_test}")

for image_size in image_sizes:
    for epoch_size in epochs:
        for batch_size in batch_sizes:

            if yolov5.exists:
                # YOLO V5
                for tested_dataset in v5_datasets_to_test:
                    for tested_model in tested_yolov5_models:
                        try:
                            logger.info(f"YOLOv5 training starting for: "
                                        f"Image size: {image_size}, "
                                        f"Epoch size: {epoch_size}, "
                                        f"Batch size: {batch_size}, "
                                        f"Dataset: {tested_dataset}, "
                                        f"Model: {tested_model}")

                            yolov5.conduct_experiments(tested_dataset,
                                                       tested_model,
                                                       image_size,
                                                       epoch_size,
                                                       batch_size)
                            logger.info(f"YOLOv5 training successfully ended for: "
                                        f"Image size: {image_size}, "
                                        f"Epoch size: {epoch_size}, "
                                        f"Batch size: {batch_size}, "
                                        f"Dataset: {tested_dataset}, "
                                        f"Model: {tested_model}")

                            logger.info(f"YOLOv5 testing starting for: "
                                        f"Image size: {image_size}, "
                                        f"Epoch size: {epoch_size}, "
                                        f"Batch size: {batch_size}, "
                                        f"Dataset: {tested_dataset}, "
                                        f"Model: {tested_model}")

                            # yolov5.conduct_testing(tested_dataset,
                            #                        tested_model,
                            #                        image_size,
                            #                        epoch_size,
                            #                        batch_size)

                            logger.info(f"YOLOv5 testing successfully ended for: "
                                        f"Image size: {image_size}, "
                                        f"Epoch size: {epoch_size}, "
                                        f"Batch size: {batch_size}, "
                                        f"Dataset: {tested_dataset}, "
                                        f"Model: {tested_model}")
                        except Exception:
                            logger.exception("Unknown error during yolov5 processing!")

            if yolov8.exists:
                # YOLO V8
                for tested_dataset in v8_datasets_to_test:
                    for tested_model in tested_yolov8_models:

                        try:
                            logger.info(f"YOLOv8 training starting for: "
                                        f"Image size: {image_size}, "
                                        f"Epoch size: {epoch_size}, "
                                        f"Batch size: {batch_size}, "
                                        f"Dataset: {tested_dataset}, "
                                        f"Model: {tested_model}")

                            yolov8.conduct_experiments(tested_dataset,
                                                       tested_model,
                                                       image_size,
                                                       epoch_size,
                                                       batch_size)

                            logger.info(f"YOLOv8 training successfully ended for: "
                                        f"Image size: {image_size}, "
                                        f"Epoch size: {epoch_size}, "
                                        f"Batch size: {batch_size}, "
                                        f"Dataset: {tested_dataset}, "
                                        f"Model: {tested_model}")

                            logger.info(f"YOLOv8 testing starting for: "
                                        f"Image size: {image_size}, "
                                        f"Epoch size: {epoch_size}, "
                                        f"Batch size: {batch_size}, "
                                        f"Dataset: {tested_dataset}, "
                                        f"Model: {tested_model}")

                            # yolov8.conduct_testing(tested_dataset,
                            #                        tested_model,
                            #                        image_size,
                            #                        epoch_size,
                            #                        batch_size)

                            logger.info(f"YOLOv8 testing successfully ended for: "
                                        f"Image size: {image_size}, "
                                        f"Epoch size: {epoch_size}, "
                                        f"Batch size: {batch_size}, "
                                        f"Dataset: {tested_dataset}, "
                                        f"Model: {tested_model}")

                        except Exception:
                            logger.exception("Unknown error during yolov8 processing!")

            if yolov6.exists:
                # YOLO V6
                for tested_dataset in v6_datasets_to_test:
                    for tested_model in tested_yolov6_models:

                        # YOLOv6 does not support auto batch size
                        if batch_size == -1:
                            continue

                        try:
                            logger.info(f"YOLOv6 training starting for: "
                                        f"Image size: {image_size}, "
                                        f"Epoch size: {epoch_size}, "
                                        f"Batch size: {batch_size}, "
                                        f"Dataset: {tested_dataset}, "
                                        f"Model: {tested_model}")

                            yolov6.conduct_experiments(tested_dataset,
                                                       tested_model,
                                                       image_size,
                                                       epoch_size,
                                                       batch_size)

                            logger.info(f"YOLOv6 training successfully ended for: "
                                        f"Image size: {image_size}, "
                                        f"Epoch size: {epoch_size}, "
                                        f"Batch size: {batch_size}, "
                                        f"Dataset: {tested_dataset}, "
                                        f"Model: {tested_model}")

                            logger.info(f"YOLOv6 testing starting for: "
                                        f"Image size: {image_size}, "
                                        f"Epoch size: {epoch_size}, "
                                        f"Batch size: {batch_size}, "
                                        f"Dataset: {tested_dataset}, "
                                        f"Model: {tested_model}")

                            yolov6.conduct_testing(tested_dataset,
                                                   tested_model,
                                                   image_size,
                                                   epoch_size,
                                                   batch_size)

                            logger.info(f"YOLOv6 testing successfully ended for: "
                                        f"Image size: {image_size}, "
                                        f"Epoch size: {epoch_size}, "
                                        f"Batch size: {batch_size}, "
                                        f"Dataset: {tested_dataset}, "
                                        f"Model: {tested_model}")
                        except Exception:
                            logger.exception("Unknown error during yolov6 processing!")

            if yolov7.exists:
                # YOLO V7
                for tested_dataset in v7_datasets_to_test:
                    for tested_model in tested_yolov7_models:

                        # YOLOv7 does not support auto batch size
                        if batch_size == -1:
                            continue

                        try:
                            logger.info(f"YOLOv7 training starting for: "
                                        f"Image size: {image_size}, "
                                        f"Epoch size: {epoch_size}, "
                                        f"Batch size: {batch_size}, "
                                        f"Dataset: {tested_dataset}, "
                                        f"Model: {tested_model}")

                            yolov7.conduct_experiments(tested_dataset,
                                                       tested_model,
                                                       image_size,
                                                       epoch_size,
                                                       batch_size)

                            logger.info(f"YOLOv7 training successfully ended for: "
                                        f"Image size: {image_size}, "
                                        f"Epoch size: {epoch_size}, "
                                        f"Batch size: {batch_size}, "
                                        f"Dataset: {tested_dataset}, "
                                        f"Model: {tested_model}")

                            logger.info(f"YOLOv7 testing starting for: "
                                        f"Image size: {image_size}, "
                                        f"Epoch size: {epoch_size}, "
                                        f"Batch size: {batch_size}, "
                                        f"Dataset: {tested_dataset}, "
                                        f"Model: {tested_model}")

                            # yolov7.conduct_testing(tested_dataset,
                            #                        tested_model,
                            #                        image_size,
                            #                        epoch_size,
                            #                        batch_size)

                            logger.info(f"YOLOv7 testing successfully ended for: "
                                        f"Image size: {image_size}, "
                                        f"Epoch size: {epoch_size}, "
                                        f"Batch size: {batch_size}, "
                                        f"Dataset: {tested_dataset}, "
                                        f"Model: {tested_model}")
                        except Exception:
                            logger.exception("Unknown error during yolov7 processing!")

# Uncomment if you want to shut down the computer after experiments
#os.system('shutdown now -h')
