# Pano Scroller

A graphical app, allowing users to improve datasets of equirectangular panoramas.
It may be successfully used in the image classification task (works similarily to the horizontal translation available in Keras library as a Sequential augmentation layer)
or in object detection - by augmenting and changing the annotations.

# Publications
This project was a part of processing used in the publication **"Finding the perfect cut: selection of the best cutting point in equirectangular panoramas for object detection"** - https://doi.org/10.1016/j.procs.2024.09.432 
Presented at the 28th International Conference on Knowledge-Based and Intelligent Information & Engineering Systems (KES 2024)

BibTeX citation:
 
```
@article{POKUCINSKI2024519,
title = {Finding the Perfect Cut: Selection of the Best Cutting Point in Equirectangular Panoramas for Object Detection},
journal = {Procedia Computer Science},
volume = {246},
pages = {519-528},
year = {2024},
note = {28th International Conference on Knowledge Based and Intelligent information and Engineering Systems (KES 2024)},
issn = {1877-0509},
doi = {https://doi.org/10.1016/j.procs.2024.09.432},
url = {https://www.sciencedirect.com/science/article/pii/S1877050924024724},
author = {Sebastian Pokuciński and Katarzyna Filus and Dariusz Mrozek},
keywords = {Equirectangular images, Object detection, YOLO training optimization, Panorama Splitting, Transparent testing},
abstract = {Analyzing equirectangular imagery becomes increasingly important due to its expanding utilization in numerous domains. With their wide field of view and unique format, they create novel opportunities and challenges for object detection algorithms. Training on rectilinear data, which is better accessible than panoramic data, results in low accuracy on equirectangular images. Therefore, it is important to propose new dataset preparation methodologies that allow to fully utilize the potential of the size-constrained, relatively unpopular and niche equirectangular datasets. In this paper, we proposed a novel method for the selection of the Best Cutting Point that can be used to create datasets aimed at facilitating the model’s training and testing. This approach is used to choose the meridian from which the sphere is unfolded during the transformation from a 3D spatial figure into a 2D fat image. Our target is to eliminate the disadvantages of the existing cutting methods: the time-consuming, labor-intensive, and subjective character of a manual technique and frequent distortions in the spatial placement of object features common for a random cutting method. It does that by automatically finding the best cutting point along the horizontal axis to preserve the integrity of objects relevant to the detection task, either keeping them intact or minimizing losses. To evaluate the method, we use an available dataset and also propose a new dataset called EquiB&B. The numerical results show that the datasets created with the proposed approach can facilitate the training of detection models for equirectangular data. They also show the potential of the method for building practical testing methodologies involving best- and worst-case scenarios to introduce more explainability and transparency to testing.}
}
```
