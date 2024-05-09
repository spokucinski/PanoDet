# Intro
"What's in My Room? Object Recognition on Indoor Panoramic Images" is a project and article conducted by Guerrero-Viu et al. 
The proposed dataset consists of 666 panoramas taken from the SUN360 dataset.

# Description
As a part of conducted research, there was a need for a YOLO-compatible annotated dataset for object detection.
WiMR as a similar project has introduced valuable dataset. Nevertheless, its format was not correctly read neither by online platform, nor the default implementation of YOLO-based detectors.

In this sub-project we deal with the parsing of the available data. Implemented code reads, maps and exports the same object detection annotations but in a format strictly following the official standards. As a result, parsed annotations are fully usable in the online platform of Roboflow and can be further utilized for YOLO-based networks training.

# References
For further details, please, refer to the source project: What’s in my Room? Object Recognition on Indoor Panoramic Images

# Publications
This project was a part of processing used in the publication **"Object Detection with YOLOv5 in Indoor Equirectangular Panoramas"** - https://doi.org/10.1016/j.procs.2023.10.233
Presented at the 27th International Conference on Knowledge-Based and Intelligent Information & Engineering Systems (KES 2023)

BibTeX citation:
 
```
@article{POKUCINSKI20232420,
  title = {Object Detection with YOLOv5 in Indoor Equirectangular Panoramas},
  journal = {Procedia Computer Science},
  volume = {225},
  pages = {2420-2428},
  year = {2023},
  note = {27th International Conference on Knowledge Based and Intelligent Information and Engineering Sytems (KES 2023)},
  issn = {1877-0509},
  doi = {https://doi.org/10.1016/j.procs.2023.10.233},
  url = {https://www.sciencedirect.com/science/article/pii/S1877050923013911},
  author = {Sebastian Pokuciński and Dariusz Mrozek},
  keywords = {object detection, YOLOv5, equirectangular panorama, smart indoor systems},
  abstract = {With growing interest in indoor environment digitization and virtual tour creation, the acquisition of spherical images is gaining importance. Today, these images can come from a wide variety of devices, from dedicated industrial scanners through semi-professional sports cameras to smartphones equipped with the proper software. They also constitute a crucial input for smart indoor systems, such as automatic vacuum cleaners or surveillance cameras, and are essential for the construction of a building's digital twin. One of the most popular data representations used for spherical pictures is an equirectangular panorama (ERP). Although popular, processing it is demanding - ERP is heavily distorted, typically of high resolution, and made with discontinued lighting conditions. Transferring it to the cloud is challenging and the in-place analysis on the edge device is complicated. Nevertheless, ERPs store massive amounts of data, especially in the context of the number of objects captured in a single image. Many techniques dedicated to aberration-resilient object detection were presented. However, they all required proper preprocessing of the data or changed the detection approach so much that it made them unfit for any other task. In this paper, we present research on the processing of an indoor equirectangular panorama (I-ERP) using a family of object detectors called YOLOv5 (You Only Look Once, fifth generation). We discuss the process of adjusting an already available dataset to a newer annotation representation and test the usefulness of these detectors on such input without any changes to the detector models themselves. The results achieved indicate that the newer generation of YOLO detectors can be used with satisfactory results on the panoramic input and that the performance gap in I-ERP processing between dedicated and out-of-the-shelf solutions is closing.}
}
```
