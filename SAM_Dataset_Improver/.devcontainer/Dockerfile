FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

RUN apt-get update && \
    apt-get -y upgrade && \
    DEBIAN_FRONTEND=noninteractive apt-get -qq install ffmpeg libsm6 libxext6 -y

RUN pip install --upgrade pip && \
    pip install matplotlib && \
    pip install opencv-python && \
    pip install scipy

ADD https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth ../workspaces/PanoDet/SAM_Dataset_Improver/models/sam_vit_h_4b8939.pth