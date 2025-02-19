import os
import logging
from datetime import datetime
import subprocess
import logging
import signal
import argparse
from options import Options
import re
from external.ultralytics import RTDETR
import sys

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resultsPath", type=str, default="results", help="Where to save the results")
    parser.add_argument("--logName", type=str, default="experimentLog", help="Name for the whole experiment log file")
    parser.add_argument("--projectName", type=str, default="SPHERE_CODE55", help="Name for the project")
    parser.add_argument("--datasetsPath", type=str, default="datasets", help="Where to search for the datasets")
    parser.add_argument("--datasetDefsPath", type=str, default="datasets", help="Where to search for the .yaml files with dataset definitions")
    parser.add_argument("--datasets", type=str, nargs="+", default=[
                                                                    "ALL",
                                                                    "MIXED",
                                                                    "ITEMS",
                                                                    "NO_ITEMS",
                                                                    "REAL_TOP30"
                                                                    ], help="Which datasets to test")
    parser.add_argument("--optimizers", type=str, nargs="+", default=[
                                                                    "SGD",
                                                                    # "Adam",
                                                                    # "AdamW"
                                                                    ], help="Which optimizers to use")
    parser.add_argument("--epochs", type=int, nargs="+", default=[250], help="Training lenght in epochs")
    parser.add_argument("--patience", type=int, default=100, help="How many epochs without improvement before early stopping")
    parser.add_argument("--models", type=str, nargs="+", default=[# "yolov5s",
                                                                  "rtdetr-l"
                                                                  # "yolo11n",
                                                                  # "yolov5x",
                                                                  # "yolov5s6",
                                                                  # "yolov5m6",
                                                                  # "yolov5x6",
                                                                  ], help="What models use in training")
    parser.add_argument("--batchSizes", type=int, nargs="+", default=[3], help="Size of the batch, -1 for auto-batch")
    parser.add_argument("--imageSizes", type=int, nargs="+", default=[1792], help="Image sizes to be used in training")
    parser.add_argument("--rectangularTraining", type=bool, default=True, help="Expect the image to be rectangular, not a square")
    parser.add_argument("--hyperParameters", type=str, default="external/data/hyps/hyp.no-augmentation.yaml", help="Path to hyperparameters configuration .yaml file")

    return parser.parse_args()

def loadDoneExperiments(resultsPath: str, projectName: str, experimentType: str = "Train") -> list[str]:
    
    expectedResultsPath = os.path.join(resultsPath, projectName, experimentType)
    
    if not os.path.exists(expectedResultsPath):
        os.makedirs(expectedResultsPath)

    return [doneTrainingConfiguration for doneTrainingConfiguration \
            in os.listdir(expectedResultsPath) \
            if os.path.isdir(os.path.join(expectedResultsPath, doneTrainingConfiguration))]

def loadDatasets(datasetsPath: str) -> list[str]:
    
    if not os.path.exists(datasetsPath):
        os.makedirs(datasetsPath)
    
    return [dataset for dataset 
            in os.listdir(datasetsPath) 
            if os.path.isdir(os.path.join(datasetsPath, dataset))]

def initializeExperimentLogger(options: Options):
    expDate:str = datetime.now().strftime("%Y%m%d%H%M%S")
    
    logging.basicConfig(level=logging.INFO,
                        filename=os.path.join(options.resultsPath, options.logName + f'{expDate}.log'),
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logger = logging.getLogger("ExperimentsRunner")

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    consoleHandler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(consoleHandler)

    logger.info("Starting experiments!")
    logger.info("Reading start parameters...")
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

def conductTesting(imageSize: int, 
                   batchSize: int, 
                   runName: str, 
                   modelPath: str, 
                   dataDefPath: str, 
                   alreadyConductedTests: list[str],
                   resultsPath: str,
                   projectName: str):
    
    if runName not in alreadyConductedTests:
        try:
            model = RTDETR(modelPath)
            
            testLogDir = f'{resultsPath}/{projectName}/Test/{runName}'       
            if not os.path.exists(testLogDir):
                os.makedirs(testLogDir)
            testLogPath = f'{testLogDir}/testLog.txt'

            with open(testLogPath, 'w') as f:
                sys.stdout = f
                print(f'TESTING LOG OF: {runName}')
                results = model.val(split='test', 
                    imgsz=imageSize, 
                    data=dataDefPath, 
                    name=runName, 
                    project=f'{resultsPath}/{projectName}/Test',
                    save_json=True,
                    batch=1,
                    rect=True,
                    plots=True)
                print(f"map50: {results.results_dict['metrics/mAP50(B)']}")
                print(f"map50-95: {results.results_dict['metrics/mAP50-95(B)']}")
            sys.stdout = sys.__stdout__
            # clean_file(testLogPath)

        except Exception as testingException:
            logger = logging.getLogger(__name__)
            logger.error(f"Exception during YOLOv5 testing!")
            logger.error(f"Killing YOLOv5 testing!")
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            logger.error("YOLOv5 testing killed!")

def clean_file(filename):
    # Read the content of the file
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove ANSI escape sequences
    clean_content = re.sub(r'\x1b\[[0-9;]*[mGK]', '', content)

    # Write the cleaned content back to the file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(clean_content)

def conductTraining(epochs: int, 
                    imageSize: int, 
                    batchSize: int, 
                    runName: str, 
                    modelPath: str, 
                    dataDefPath: str, 
                    alreadyConductedTrainings: list[str],
                    resultsPath: str,
                    projectName: str,
                    hyperParPath: str,
                    rect: bool,
                    optimizer: str):
    
    if runName not in alreadyConductedTrainings:
        try:
            train_cmd = ['yolo', 'detect','train',
                        f'epochs={epochs}',
                        f'imgsz={imageSize}',
                        f'batch={batchSize}',
                        f'model={modelPath}', #external/{model}.pt',
                        f'data={dataDefPath}', #{options.datasetDefsPath}/{dataset}.yaml',
                        f'name={runName}',
                        f'project={resultsPath}/{projectName}/Train',
                        #f'hyp={hyperParPath}', #external/data/hyps/hyp.no-augmentation.yaml'
                        f'exist_ok=True',
                        #f'optimizer={optimizer}'
                        f'hsv_h=0.0',
                        f'hsv_s=0.0',
                        f'hsv_v=0.0',
                        f'degrees=0.0',
                        f'translate=0.0',
                        f'scale=0.0',
                        f'shear=0.0',
                        f'perspective=0.0',
                        f'flipud=0.0',
                        f'fliplr=0.0',
                        f'bgr=0.0',
                        f'mosaic=0.0',
                        f'mixup=0.0',
                        f'copy_paste=0.0',
                        f'erasing=0.0',
                        f'crop_fraction=1.0',
                        ]
            
            if rect:
                train_cmd.append(f'rect=True')
            
            trainLogDir = f'{resultsPath}/{projectName}/Train/{runName}'
            if not os.path.exists(trainLogDir):
                os.makedirs(trainLogDir)
            
            trainLogPath = f'{trainLogDir}/trainLog.txt'       
            trainLog = open(trainLogPath, 'a')
            trainLog.write(f'TRAINING LOG OF: {runName}')
            trainLog.flush()
            p = subprocess.Popen(train_cmd, stdout=trainLog, stderr=trainLog, start_new_session=True)
            p.wait(timeout=250000)

            clean_file(trainLogPath)
        
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Exception during YOLOv5 training! Run command: {train_cmd}")
            logger.error(f"Killing YOLOv5 training!")
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            logger.error("YOLOv5 training killed!")

def main(options: Options):

    datasetsFound: list[str] = loadDatasets(options.datasetsPath)

    alreadyConductedTrainings = loadDoneExperiments(options.resultsPath, options.projectName, "Train")
    alreadyConductedTests = loadDoneExperiments(options.resultsPath, options.projectName, "Test")

    logger = initializeExperimentLogger(options)

    for dataset in options.datasets:
        if dataset not in datasetsFound:
            logger.error("Requested dataset: {dataset} not found!")
            continue
        
        for model in options.models:    
            for epochNum in options.epochs:
                for imageSize in options.imageSizes:
                    for batchSize in options.batchSizes:
                        for optimizer in options.optimizers:
                            dataDefPath = f'{options.datasetDefsPath}/{dataset}/dataset.yaml'
                            modelPath = f'{model}.pt' #f'external/{model}.pt'
                            runConfiguration: str = f"{dataset}_{epochNum}_{imageSize}_{batchSize}_{model}_{optimizer}"                     
                            
                            # conductTraining(epochNum, 
                            #                 imageSize, 
                            #                 batchSize, 
                            #                 runConfiguration, 
                            #                 modelPath, 
                            #                 dataDefPath, 
                            #                 alreadyConductedTrainings, 
                            #                 options.resultsPath, 
                            #                 options.projectName, 
                            #                 options.hyperParameters,
                            #                 options.rectangularTraining,
                            #                 optimizer)
                            
                            # bestTrainingModelPath = f'{options.resultsPath}/{options.projectName}/Train/{runConfiguration}/weights/best.pt'

                            bestTrainingModelPath = 'results/SPHERE_CODE55/Train/REAL_TOP30_250_1792_3_rtdetr-l_SGD/weights/best.pt'           

                            conductTesting(imageSize, 
                                        1, 
                                        runConfiguration, 
                                        bestTrainingModelPath, 
                                        dataDefPath, 
                                        alreadyConductedTests, 
                                        options.resultsPath, 
                                        options.projectName)

if __name__ == "__main__":
    opt = parse_opt()
    trainingOptions: Options = \
        Options(opt.resultsPath,
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
                opt.hyperParameters,
                opt.optimizers)
    main(trainingOptions)