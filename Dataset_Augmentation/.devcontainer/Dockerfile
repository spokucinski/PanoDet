FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

RUN apt-get update && \
    apt-get -y upgrade

RUN pip install --upgrade pip && \
    pip install matplotlib && \
    pip install opencv-python && \
    pip install scipy && \
    pip install progressbar && \
    pip install gdown && \
    pip install tqdm && \
    pip install -U albumentations