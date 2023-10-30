import os
import gdown
from zipfile import ZipFile
from tqdm import tqdm

SAM_MODEL_URL : str = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
SAM_MODEL_FILENAME : str = 'sam_vit_h_4b8939.pth'
SAM_MODEL_DEST_PATH : str = os.path.join('models', SAM_MODEL_FILENAME)

PANDORA_DATASET_URL : str = 'https://drive.google.com/uc?id=1JAGReczN_h3F3mY-mlGTVSeDx-CCJigC'
PANDORA_DATASET_FILENAME : str = 'pandora.zip'
PANDORA_DATASET_DEST_DIR : str = os.path.join('data', 'input')
PANDORA_DATASET_DEST_ZIP_PATH : str = os.path.join(PANDORA_DATASET_DEST_DIR, PANDORA_DATASET_FILENAME)

gdown.download(SAM_MODEL_URL, SAM_MODEL_DEST_PATH, quiet=False)
gdown.download(PANDORA_DATASET_URL, PANDORA_DATASET_DEST_ZIP_PATH, quiet=False)

with ZipFile(PANDORA_DATASET_DEST_ZIP_PATH, 'r') as archive:
    for member in tqdm(archive.infolist(), desc='Extracting '):
        try:
            archive.extract(member, PANDORA_DATASET_DEST_DIR)
        except archive.error as e:
            pass
os.remove(PANDORA_DATASET_DEST_ZIP_PATH)
    