import os
import logging
from datetime import datetime
import subprocess
import logging
import signal
import argparse
from options import TrainingOptions

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resultsPath", type=str, default="results", help="Where to save the results")
    parser.add_argument("--logName", type=str, default="experimentLog", help="Name for the whole experiment log file")
    parser.add_argument("--projectName", type=str, default="PanoDet", help="Name for the project")
    parser.add_argument("--datasetsPath", type=str, default="datasets", help="Where to search for the datasets")
    parser.add_argument("--datasetDefsPath", type=str, default="data", help="Where to search for the .yaml files with dataset definitions")
    parser.add_argument("--epochs", type=int, nargs="+", default=[500], help="Training lenght in epochs")
    parser.add_argument("--patience", type=int, default=100, help="How many epochs without improvement before early stopping")
    parser.add_argument("--models", type=str, nargs="+", default=['yolov5x', 'yolov5m'], help="What models use in training")
    parser.add_argument("--batchSizes", type=int, nargs="+", default=[-1], help="Size of the batch, -1 for auto-batch")
    parser.add_argument("--imageSizes", type=int, nargs="+", default=[1024], help="Image sizes to be used in training")
    parser.add_argument("--rectangularTraining", type=bool, default=True, help="Expect the image to be rectangular, not a square")
    parser.add_argument("--hyperParameters", type=str, default="external/data/hyps/hyp.no-augmentation.yaml", help="Path to hyperparameters configuration .yaml file")

    return parser.parse_args()

def main(options: TrainingOptions):

    exp_date:str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    datasets_found: list[str] = [dataset for dataset in os.listdir(options.datasetsPath) 
                                if os.path.isdir(os.path.join(options.datasetsPath, dataset))]

    if not os.path.exists(os.path.join(options.resultsPath, options.projectName, 'Train')):
        os.makedirs(os.path.join(options.resultsPath, options.projectName, 'Train'))

    alreadyConductedTrainings: list[str] = ["_".join(doneTrainingConfiguration.split('_')[:-2]) \
                                            for doneTrainingConfiguration \
                                                in os.listdir(os.path.join(options.resultsPath, options.projectName, 'Train')) \
                                                    if os.path.isdir(os.path.join(options.resultsPath, options.projectName, 'Train', doneTrainingConfiguration))]

    logging.basicConfig(level=logging.INFO,
                        filename=os.path.join(options.resultsPath, options.logName + f'{exp_date}.log'),
                        filemode='a',
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger = logging.getLogger("ExperimentsRunner")

    # # Get YOLOv5 logger and add custom file handler
    # logger = val.LOGGER
    # # Create a file handler to log messages to a file
    # filePath = os.path.join(RES_PATH, LOG_NAME + f'{exp_date}.log')
    # directory = os.path.dirname(filePath)
    # # Check if the directory exists, and create it if it doesn't
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # file_handler = logging.FileHandler(filePath)
    # file_handler.setLevel(logging.INFO)  # Set the logging level for this handler
    # # Optionally, set a formatter
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    logger.info("Starting experiments!")
    logger.info("Reading start parameters...")
    logger.info("Requested experiment parameters:")
    logger.info(f'Project name: {options.projectName}')
    logger.info(f'Results path: {options.resultsPath}')
    logger.info(f'Datasets will be searched for in: {options.datasetsPath}')
    logger.info(f'Epochs: {options.epochs}')
    logger.info(f'Image sizes: {options.imageSizes}')
    logger.info(f'Batch sizes: {options.batchSizes}')
    logger.info(f'Models: {options.models}')

    for model in options.models:
        for dataset in datasets_found:
            for epochNum in options.epochs:
                for imageSize in options.imageSizes:
                    for batchSize in options.batchSizes:

                        runConfiguration: str = f"{dataset}_{epochNum}_{imageSize}_{batchSize}_{model}"
                        if runConfiguration in alreadyConductedTrainings:
                            continue
                        
                        runName:str = f"{runConfiguration}_{exp_date}"
                        
                        try:
                            train_cmd = ['python', 'external/train.py',
                                        f'--epochs={epochNum}',
                                        f'--imgsz={imageSize}',
                                        f'--batch-size={batchSize}',
                                        f'--weights=external/{model}.pt',
                                        f'--rect',
                                        f'--data={options.datasetDefsPath}/{dataset}.yaml',
                                        f'--name={runName}',
                                        f'--project={options.resultsPath}/{options.projectName}/Train',
                                        f'--hyp=external/data/hyps/hyp.no-augmentation.yaml'
                                        ]
                            
                            train_path = f'{options.resultsPath}/YOLOv5/{dataset}/TrainLog/{runName}'
                            if not os.path.exists(train_path):
                                os.makedirs(train_path)
                            train_log = open(
                                f'{options.resultsPath}/YOLOv5/{dataset}/TrainLog/{runName}/run_log.txt', 'a')
                            train_log.write(f'TRAINING LOG OF: {runName}')
                            train_log.flush()
                            p = subprocess.Popen(train_cmd, stdout=train_log, stderr=train_log, start_new_session=True)
                            p.wait(timeout=14400)
                            # opt = train.parse_opt()
                            # opt.epochs = epochNum
                            # opt.imgsz = imageSize
                            # opt.batch_size = batchSize
                            # opt.weights = PosixPath(f'external/{model}.pt')
                            # opt.rect = True
                            # opt.data = PosixPath(f'{DATA_PATH}/{dataset}.yaml')
                            # opt.name = runName
                            # opt.project = PosixPath(f'{RES_PATH}/{PROJ_NAME}/Train')
                            # opt.hyp = PosixPath('external/data/hyps/hyp.no-augmentation.yaml')
                            # train.main(opt)

                        except Exception as e:
                            logger = logging.getLogger(__name__)
                            logger.error(f"Exception during YOLOv5 testing! Run command: {train_cmd}")
                            logger.error(f"Killing YOLOv5 testing!")
                            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                            logger.error("YOLOv5 testing killed!")

                        try:

                            test_cmd = ['python', 'external/val.py',
                                        f'--task=test',
                                        f'--imgsz={imageSize}',
                                        f'--weights={options.resultsPath}/{options.projectName}/Train/{runName}/weights/best.pt',
                                        f'--data={options.datasetDefsPath}/{dataset}.yaml',
                                        f'--name={runName}',
                                        f'--project={options.resultsPath}/{options.projectName}/Train',
                                        f'--verbose',
                                        f'--save_txt',
                                        f'--batch-size=1',
                                        ]
                            
                            test_path = f'{options.resultsPath}/YOLOv5/{dataset}/TestLog/{runName}'
                            if not os.path.exists(test_path):
                                os.makedirs(test_path)
                            test_log = open(f'{options.resultsPath}/YOLOv5/{dataset}/TestLog/{runName}/run_log.txt', 'a')
                            test_log.write(f'TESTING LOG OF: {runName}')
                            test_log.flush()
                            p = subprocess.Popen(test_cmd, stdout=test_log, stderr=test_log, start_new_session=True)
                            p.wait(timeout=14400)
                            # testOpt = val.parse_opt()
                            # testOpt.data = PosixPath(f'{DATA_PATH}/{dataset}.yaml')
                            # testOpt.weights = PosixPath(f'{RES_PATH}/{PROJ_NAME}/Train/{runName}/weights/best.pt')
                            # testOpt.batch_size = 1
                            # testOpt.imgsz = imageSize
                            # testOpt.save_txt = True
                            # testOpt.verbose = True
                            # testOpt.task = 'test'
                            # testOpt.name = runName
                            # testOpt.project = PosixPath(f'{RES_PATH}/{PROJ_NAME}/Test')
                            
                            # val.main(testOpt)

                        except Exception as testingException:
                            logger = logging.getLogger(__name__)
                            logger.error(f"Exception during YOLOv5 testing! Run command: {test_cmd}")
                            logger.error(f"Killing YOLOv5 testing!")
                            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                            logger.error("YOLOv5 testing killed!")

                            #logger.exception('Error during testing!', testingException)


if __name__ == "__main__":
    opt = parse_opt()
    trainingOptions: TrainingOptions = \
        TrainingOptions(opt.resultsPath,
                        opt.logName,
                        opt.projectName,
                        opt.datasetsPath,
                        opt.datasetDefsPath,
                        opt.epochs,
                        opt.patience,
                        opt.models,
                        opt.batchSizes,
                        opt.imageSizes,
                        opt.rectangularTraining,
                        opt.hyperParameters)
    main(trainingOptions)