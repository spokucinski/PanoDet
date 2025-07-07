import os
import logging
import argparse
from datetime import datetime
import subprocess
import signal
import re
from typing import List

class Options:
    def __init__(self, results_path, log_name, project_name, datasets_path,
                 dataset_defs_path, datasets, epochs, patience, models,
                 batch_sizes, image_sizes, rectangular_training, optimizers, augmentation_modes):
        self.resultsPath = results_path
        self.logName = log_name
        self.projectName = project_name
        self.datasetsPath = datasets_path
        self.datasetDefsPath = dataset_defs_path
        self.datasets = datasets
        self.epochs = epochs
        self.patience = patience
        self.models = models
        self.batchSizes = batch_sizes
        self.imageSizes = image_sizes
        self.rectangularTraining = rectangular_training
        self.optimizers = optimizers
        self.augmentationModes = augmentation_modes

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resultsPath", type=str, default="results")
    parser.add_argument("--logName", type=str, default="experimentLog")
    parser.add_argument("--projectName", type=str, default="FullCODE55")
    parser.add_argument("--datasetsPath", type=str, default="datasets")
    parser.add_argument("--datasetDefsPath", type=str, default="datasets")
    parser.add_argument("--datasets", type=str, nargs="+", default=["FullCode", "ScrolledFullCode"])
    parser.add_argument("--optimizers", type=str, nargs="+", default=["auto"]) # Possibly worth to go for other optimizers
    parser.add_argument("--epochs", type=int, nargs="+", default=[500])
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--models", type=str, nargs="+", default=["yolo11x"])
    parser.add_argument("--batchSizes", type=int, nargs="+", default=[3])
    parser.add_argument("--imageSizes", type=int, nargs="+", default=[1920]) # Possibly worth to go for twice as much -> half of training images are in higher resolution
    parser.add_argument("--rectangularTraining", type=bool, default=True)
    parser.add_argument("--augmentationModes", type=str, nargs="+", default=["default"],
                        choices=["default", "none", "custom"],
                        help="Augmentation modes to run: default, none, custom")
    
    return parser.parse_args()

def initialize_logger(options: Options):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    os.makedirs(options.resultsPath, exist_ok=True)
    log_file = os.path.join(options.resultsPath, f"{options.logName}_{timestamp}.log")

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])
    
    logger = logging.getLogger("Experiments")

    logger.info("Experiment session started")
    logger.info("Requested experiment parameters:")
    logger.info(f'Project name: {options.projectName}')
    logger.info(f'Results path: {options.resultsPath}')
    logger.info(f'Datasets will be searched for in: {options.datasetsPath}')
    logger.info(f'Datasets to be experimented with: {options.datasets}')
    logger.info(f'Epochs: {options.epochs}')
    logger.info(f'Image sizes: {options.imageSizes}')
    logger.info(f'Batch sizes: {options.batchSizes}')
    logger.info(f'Models: {options.models}')

    return logger

def load_existing_runs(results_path: str, project_name: str, experiment_type: str) -> List[str]:
    path = os.path.join(results_path, project_name, experiment_type)
    os.makedirs(path, exist_ok=True)
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def load_datasets(datasets_path: str) -> List[str]:
    os.makedirs(datasets_path, exist_ok=True)
    return [d for d in os.listdir(datasets_path) if os.path.isdir(os.path.join(datasets_path, d))]

def clean_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = re.sub(r'\x1b\[[0-9;]*[mGK]', '', f.read())
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def run_subprocess(command, log_path):
    with open(log_path, 'w') as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=log_file, start_new_session=True)
        try:
            process.wait(timeout=250000)
        except Exception:
            logging.error("Subprocess error, killing process...")
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        finally:
            clean_file(log_path)

def train_model(opt: Options, run_name: str, model: str, data_path: str,
                img_size: int, batch_size: int, epochs: int, optimizer: str, augmentation_mode: str):
    
    output_dir = os.path.join(opt.resultsPath, opt.projectName, "Train", run_name)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "trainLog.txt")

    cmd = [
        "yolo", "train",
        f"model={model}",
        f"data={data_path}",
        f"imgsz={img_size}",
        f"batch={batch_size}",
        f"epochs={epochs}",
        f"name={run_name}",
        f"project={os.path.join(opt.resultsPath, opt.projectName, 'Train')}",
        f"exist_ok=True",
        f"patience={opt.patience}",
        f"optimizer={optimizer}"
    ]

    if opt.rectangularTraining:
        cmd.append("rect=True")
    
    if augmentation_mode == "none":
        cmd += [
            "auto_augment=None",
            "hsv_h=0.0", "hsv_s=0.0", "hsv_v=0.0",
            "translate=0.0", "scale=0.0", "shear=0.0",
            "perspective=0.0", "flipud=0.0", "fliplr=0.0",
            "mosaic=0.0", "mixup=0.0", "cutmix=0.0", "copy_paste=0.0",
            "erasing=0.0", "crop_fraction=1.0"
        ]
    elif augmentation_mode == "custom":
        cmd += [
            "auto_augment=None",
            "flipud=0.0", "fliplr=0.5",
            "grid=0.2", "dropout=0.2",
            "gray=0.2", "clahe=0.2",
            "blur=0.2", "jitter=0.4",
            "sharp=0.25", "compress=0.2", "iso=0.2"
        ]

    run_subprocess(cmd, log_path)

def test_model(opt: Options, run_name: str, model_path: str, data_path: str, img_size: int):
    output_dir = os.path.join(opt.resultsPath, opt.projectName, "Test", run_name)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "testLog.txt")

    cmd = [
        "yolo", "val",
        f"model={model_path}",
        f"data={data_path}",
        f"imgsz={img_size}",
        f"batch=1",
        f"name={run_name}",
        f"project={os.path.join(opt.resultsPath, opt.projectName, 'Test')}",
        f"save_json=True",
        f"plots=True",
        "split=test"
    ]
    run_subprocess(cmd, log_path)

def main(opt: Options):
    logger = initialize_logger(opt)
    datasets = load_datasets(opt.datasetsPath)
    done_trainings = load_existing_runs(opt.resultsPath, opt.projectName, "Train")
    done_tests = load_existing_runs(opt.resultsPath, opt.projectName, "Test")

    for dataset in opt.datasets:
        if dataset not in datasets:
            logger.warning(f"Dataset {dataset} not found.")
            continue

        for model in opt.models:
            for epoch in opt.epochs:
                for img_size in opt.imageSizes:
                    for batch_size in opt.batchSizes:
                        for optimizer in opt.optimizers:
                            for aug_mode in opt.augmentationModes:
                                run_name = f"{dataset}_{epoch}_{img_size}_{batch_size}_{model}_{optimizer}_{aug_mode}"
                                data_path = os.path.join(opt.datasetDefsPath, dataset, "dataset.yaml")

                                if run_name not in done_trainings:
                                    train_model(opt, run_name, model, data_path, img_size, batch_size, epoch, optimizer, aug_mode)

                                best_model = os.path.join(opt.resultsPath, opt.projectName, "Train", run_name, "weights", "best.pt")
                                if run_name not in done_tests and os.path.exists(best_model):
                                    test_model(opt, run_name, best_model, data_path, img_size)

if __name__ == "__main__":
    opt = parse_opt()
    options = Options(opt.resultsPath, 
                      opt.logName, 
                      opt.projectName,
                      opt.datasetsPath, 
                      opt.datasetDefsPath, 
                      opt.datasets,
                      opt.epochs, 
                      opt.patience, 
                      opt.models, 
                      opt.batchSizes,
                      opt.imageSizes, 
                      opt.rectangularTraining,
                      opt.optimizers,
                      opt.augmentationModes)
    main(options)
