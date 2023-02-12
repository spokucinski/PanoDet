# PanoDet
Object detection research on equirectangular panoramas

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
