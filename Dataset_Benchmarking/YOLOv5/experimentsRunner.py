import os
import logging
from datetime import datetime
from external import train
from external import val
from pathlib import PosixPath
import importlib

RES_PATH:str = 'results'
LOG_NAME:str = 'experimentLog'
PROJ_NAME:str = 'PanoDet'
DATASETS_PATH:str = 'datasets'
DATA_PATH:str = 'data'

EPOCHS_NUM:int = [500] #[500]
IMAGE_SIZES:list[int] = [1024] #[2048] #[1024, 2048, 4096]
BATCH_SIZES:list[int] = [16]
MODELS:list[str] = ['yolov5x'] #['yolov5x', 'yolov5m']

exp_date:str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
datasets_found: list[str] = [dataset for dataset in os.listdir(DATASETS_PATH) 
                             if os.path.isdir(os.path.join(DATASETS_PATH, dataset))]



# Get YOLOv5 logger and add custom file handler
logger = val.LOGGER
# Create a file handler to log messages to a file
filePath = os.path.join(RES_PATH, LOG_NAME + f'{exp_date}.log')
directory = os.path.dirname(filePath)
# Check if the directory exists, and create it if it doesn't
if not os.path.exists(directory):
    os.makedirs(directory)
file_handler = logging.FileHandler(filePath)
file_handler.setLevel(logging.INFO)  # Set the logging level for this handler
# Optionally, set a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info("Starting experiments!")
logger.info("Reading start parameters...")
logger.info("Requested experiment parameters:")
logger.info(f'Project name: {PROJ_NAME}')
logger.info(f'Results path: {RES_PATH}')
logger.info(f'Datasets will be searched for in: {DATASETS_PATH}')
logger.info(f'Epochs: {EPOCHS_NUM}')
logger.info(f'Image sizes: {IMAGE_SIZES}')
logger.info(f'Batch sizes: {BATCH_SIZES}')
logger.info(f'Models: {MODELS}')

for dataset in datasets_found:
    for epochNum in EPOCHS_NUM:
        for imageSize in IMAGE_SIZES:
            for batchSize in BATCH_SIZES:
                for model in MODELS:
                    
                    runName:str = f"{dataset}_{epochNum}_{imageSize}_{batchSize}_{model}_{exp_date}"
                    logger.warning(f"STARTING NEW EXPERIMENT: {runName}")
                    logger.warning(f"STARTING NEW EXPERIMENT: {runName}")
                    logger.warning(f"STARTING NEW EXPERIMENT: {runName}")
                    logger.warning(f"STARTING NEW EXPERIMENT: {runName}")
                    
                    try:
                        importlib.reload(train)
                        opt = train.parse_opt()
                        opt.epochs = epochNum
                        opt.imgsz = imageSize
                        opt.batch_size = batchSize
                        opt.weights = PosixPath(f'external/{model}.pt')
                        opt.rect = True
                        opt.data = PosixPath(f'{DATA_PATH}/{dataset}.yaml')
                        opt.name = runName
                        opt.project = PosixPath(f'{RES_PATH}/{PROJ_NAME}/Train')
                        opt.hyp = PosixPath('external/data/hyps/hyp.no-augmentation.yaml')
                        train.main(opt)

                    except Exception as e:
                        logger.exception('Error during training!', e)

                    # try:             
                    #     testOpt = val.parse_opt()
                    #     testOpt.data = PosixPath(f'{DATA_PATH}/{dataset}.yaml')
                    #     testOpt.weights = PosixPath(f'{RES_PATH}/{PROJ_NAME}/Train/{runName}/weights/best.pt')
                    #     testOpt.batch_size = 1
                    #     testOpt.imgsz = imageSize
                    #     testOpt.save_txt = True
                    #     testOpt.verbose = True
                    #     testOpt.task = 'test'
                    #     testOpt.name = runName
                    #     testOpt.project = PosixPath(f'{RES_PATH}/{PROJ_NAME}/Test')
                        
                    #     val.main(testOpt)

                    # except Exception as testingException:
                    #     logger.exception('Error during testing!', testingException)