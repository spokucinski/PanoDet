# PanoDet

**Computer Vision** and **Artificial Intelligence** in the context of omnidirectional, equirectangular panoramas. A summarized repository, combining all the research, results, failures and successes connected to the PhD studies conducted at the Silesian University of Technology, Gliwice, Poland.

## Repo Content

All the conducted experiments and research consists of **(but is not limited to!)**:
- [CODE_55](CODE_55/README.md) - 55 Common Objects in Domestic Environment - A novel dataset definition for the most commonly expected objects in an indoor home environment
- [CODE_MTNN](CODE_MTNN/README.md) - Multitask Neural Network try-out combining the object detection and room classification for a home-digitalization project
- [Dataset_Augmentation](Dataset_Augmentation/README.md) - Experiments with image dataset augmentations save for equirectangular panoramas
- [Dataset_Benchmarking](Dataset_Benchmarking/README.md) - YOLO-based project for experiments with panoramic datasets utilizing the scrolling mechanism
- [Dataset_Exporter](Dataset_Exporter/README.md) - Exporter of the datasets in a Roboflow-compatible format
- [Model_Exporter](Model_Exporter/README.md) - Customized exporter for the AI models to be usable in the .NET ecosystem
- [PanoScroller](Pano_Scroller/README.md) - App implementing a custom workflow for the equirectangular panoramas pre-processing
- [SAM_Dataset_Improver](SAM_Dataset_Improver/README.md) - Segment Anything Model-based dataset improver
- [WiMR_Dataset_Parser](WiMR_Dataset_Parser/README.md) - Custom made annotations parser for already existing dataset
- [YOLO_Object_Detection_Experiments](YOLO_Object_Detection_Experiments/README.md) - YOLO object detectors in the task of equirectangular panoramas processing
- [Matter_Indoor_Localization](Matter_Indoor_Localization/README.md) - Distance estimation for IoT ecosystems with Matter Protocol as coordinator

## Datasets
References and direct links for the dataset details.

> **WARNING**
> 
> Generally speaking, datasets used in the repository projects are images of a very specific type.
> They are mostly in the format of equirectangular panoramas and from the context of domestic environment.
> Because of the high concentration of user-specific personal data **the access may be limited to verified users only**.
> 
> In some cases the datasets were published a long time ago, before most of the privacy regulations and are freely available online.
> In such cases we redirect viewers to the original project sites.
>
> In some cases the datasets are novel and fresh.
> In such cases, the privacy of the people sharing the images is the top priority and I do my best to secure it.
> The images may contain manually-checked pictures with blurred areas, covering critically private parts.
> 
> Images present a private indoor domestic environment - it should stay private and the use of collected pictures is expected to be strictly monitored.

### WiMR Dataset [Existing Project, Link]
"What's in My Room? Object Recognition on Indoor Panoramic Images" is a project and article conducted by Guerrero-Viu et al. The dataset consists of 666 panoramas taken from the SUN360 dataset.

For further details, please, refer to the source project:
[What’s in my Room? Object Recognition on Indoor Panoramic Images](https://webdiis.unizar.es/~jguerrer/room_OR/)

### PANDORA Dataset [Existing Project, Link]
"PANDORA: A Panoramic Detection Dataset for Object with Orientation" is a project and article conducted by Xu et al. The dataset is a part of project called "Spherical Object Detection". 

For further details, please, refer to the project's GitHub page:
[SphericalObjectDetection](https://github.com/tdsuper/SphericalObjectDetection)

### Apartment Air B&B [Novel, Monitored Access]
The "Apartment Air B&B" dataset is available under the following Google Drive's link:
[Limited Access Link](https://drive.google.com/file/d/1_JgxNAFwwSJ7MblE26UYsYJoJBrUQqWr/view?usp=sharing)

### SPHEREA [Novel, Monitored Access]
Synthetic images of indoor environments generated with user-defined requirements and structured generation prompts.
[Limited Access Link](https://drive.google.com/drive/folders/1Qrleo5YZe3HnNLu7zDff7hK_f8dAyAKk?usp=sharing)
