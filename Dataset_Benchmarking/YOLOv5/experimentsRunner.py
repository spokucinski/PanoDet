import os
import logging
from datetime import datetime
import subprocess
import logging
import signal
import argparse
from options import Options
import re

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resultsPath", type=str, default="results", help="Where to save the results")
    parser.add_argument("--logName", type=str, default="experimentLog", help="Name for the whole experiment log file")
    parser.add_argument("--projectName", type=str, default="CODE", help="Name for the project")
    parser.add_argument("--datasetsPath", type=str, default="datasets", help="Where to search for the datasets")
    parser.add_argument("--datasetDefsPath", type=str, default="data", help="Where to search for the .yaml files with dataset definitions")
    parser.add_argument("--datasets", type=str, nargs="+", default=[
                                                                    "AutomaticallyMaxScrolledDataset",
                                                                    "AutomaticallyMinScrolledDataset",
                                                                    "ManuallyMaxScrolledDataset",
                                                                    "ManuallyMinScrolledDataset",
                                                                    "UnscrolledDataset",
                                                                    "UnifiedDistributionDataset1", 
                                                                    "UnifiedDistributionDataset2",
                                                                    "UnifiedDistributionDataset3",
                                                                    "UnifiedDistributionDataset4",
                                                                    "UnifiedDistributionDataset5"
                                                                    # "AutomaticallyMaxScrolledPanoDet",
                                                                    # "AutomaticallyMinScrolledPanoDet",
                                                                    # "ManuallyMaxScrolledPanoDet",
                                                                    # "ManuallyMinScrolledPanoDet",
                                                                    # "UnscrolledPanoDet",
                                                                    # "UnifiedDistributionPanoDet1", 
                                                                    # "UnifiedDistributionPanoDet2",
                                                                    # "UnifiedDistributionPanoDet3",
                                                                    # "UnifiedDistributionPanoDet4",
                                                                    # "UnifiedDistributionPanoDet5"
                                                                    ], help="Which datasets to test")
    parser.add_argument("--epochs", type=int, nargs="+", default=[500], help="Training lenght in epochs")
    parser.add_argument("--patience", type=int, default=100, help="How many epochs without improvement before early stopping")
    parser.add_argument("--models", type=str, nargs="+", default=['yolov5n'], help="What models use in training")
    parser.add_argument("--batchSizes", type=int, nargs="+", default=[4], help="Size of the batch, -1 for auto-batch")
    parser.add_argument("--imageSizes", type=int, nargs="+", default=[2048], help="Image sizes to be used in training")
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
            test_cmd = ['python', 'external/val.py',
                        f'--task=test',
                        f'--imgsz={imageSize}',
                        f'--weights={modelPath}', #{options.resultsPath}/{options.projectName}/Train/{runName}/weights/best.pt',
                        f'--data={dataDefPath}', #{options.datasetDefsPath}/{dataset}.yaml',
                        f'--name={runName}',
                        f'--project={resultsPath}/{projectName}/Test',
                        f'--verbose',
                        f'--save-txt',
                        f'--batch-size={batchSize}',
                        f'--exist-ok'
                        ]
            
            testLogDir = f'{resultsPath}/{projectName}/Test/{runName}'       
            if not os.path.exists(testLogDir):
                os.makedirs(testLogDir)
                
            testLogPath = f'{testLogDir}/testLog.txt'
            testLogFile = open(testLogPath, 'a')
            testLogFile.write(f'TESTING LOG OF: {runName}')
            testLogFile.flush()
            p = subprocess.Popen(test_cmd, stdout=testLogFile, stderr=testLogFile, start_new_session=True)
            p.wait(timeout=25000)

            clean_file(testLogPath)

        except Exception as testingException:
            logger = logging.getLogger(__name__)
            logger.error(f"Exception during YOLOv5 testing! Run command: {test_cmd}")
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
                    hyperParPath: str):
    
    if runName not in alreadyConductedTrainings:
        try:
            train_cmd = ['python', 'external/train.py',
                        f'--epochs={epochs}',
                        f'--imgsz={imageSize}',
                        f'--batch-size={batchSize}',
                        f'--weights={modelPath}', #external/{model}.pt',
                        f'--rect',
                        f'--data={dataDefPath}', #{options.datasetDefsPath}/{dataset}.yaml',
                        f'--name={runName}',
                        f'--project={resultsPath}/{projectName}/Train',
                        f'--hyp={hyperParPath}', #external/data/hyps/hyp.no-augmentation.yaml'
                        f'--exist-ok'
                        ]
            
            trainLogDir = f'{resultsPath}/{projectName}/Train/{runName}'
            if not os.path.exists(trainLogDir):
                os.makedirs(trainLogDir)
            
            trainLogPath = f'{trainLogDir}/trainLog.txt'       
            trainLog = open(trainLogPath, 'a')
            trainLog.write(f'TRAINING LOG OF: {runName}')
            trainLog.flush()
            p = subprocess.Popen(train_cmd, stdout=trainLog, stderr=trainLog, start_new_session=True)
            p.wait(timeout=25000)

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
                        
                        dataDefPath = f'{options.datasetDefsPath}/{dataset}.yaml'
                        modelPath = f'external/{model}.pt'
                        runConfiguration: str = f"{dataset}_{epochNum}_{imageSize}_{batchSize}_{model}"                     
                        
                        # conductTraining(epochNum, 
                        #                 imageSize, 
                        #                 batchSize, 
                        #                 runConfiguration, 
                        #                 modelPath, 
                        #                 dataDefPath, 
                        #                 alreadyConductedTrainings, 
                        #                 options.resultsPath, 
                        #                 options.projectName, 
                        #                 options.hyperParameters)
                        
                        # bestTrainingModelPath = f'{options.resultsPath}/{options.projectName}/Train/{runConfiguration}/weights/best.pt'           
                        
                        bestTrainingModelPath = f'{options.resultsPath}/{options.projectName}/Train/UnifiedDistributionDataset2_500_2048_-1_yolov5n/weights/best.pt'


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
                        opt.hyperParameters)
    main(trainingOptions)