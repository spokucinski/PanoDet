# Intro
This project focuses on object detection research with use of equirectangular panoramas.

# Installation
Clone the repository to chosen folder
```
git clone https://github.com/spokucinski/PanoDet.git
```

Create a new conda environment for our project
```
conda create --name PanoDet -y
conda activate PanoDet
```

Install pip and all external requirements
```
conda install pip -y
pip install -r external/YOLOv5/requirements.txt 
pip install -r external/YOLOv6/requirements.txt 
pip install -r external/YOLOv7/requirements.txt 
pip install -r external/YOLOv8/requirements.txt
```

Install PyTorch manually one more time on top of all the auto-installs
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
```

# Publications
This part of projects was used in the experiments conducted for the paper:
**"YOLO-based object detection in panoramic images of smart buildings"** - DOI 10.1109/DSAA60987.2023.10302568
Presented at 10th International Conference on Data Science & Advanced Analytics (DSAA 2023)

Cite as BibTeX:
```
@INPROCEEDINGS{10302568,
  author={Pokuciński, Sebastian and Mrozek, Dariusz},
  booktitle={2023 IEEE 10th International Conference on Data Science and Advanced Analytics (DSAA)}, 
  title={YOLO-based Object Detection in Panoramic Images of Smart Buildings}, 
  year={2023},
  volume={},
  number={},
  pages={1-9},
  keywords={Training;Adaptation models;Smart buildings;Computational modeling;Detectors;Real-time systems;Production facilities;Equirectangular Panorama;Object Detection;Smart Indoor Environment;YOLOv5;YOLOv8},
  doi={10.1109/DSAA60987.2023.10302568}}
```
