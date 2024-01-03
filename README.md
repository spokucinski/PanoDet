Short description of the processing steps to create the dataset and baseline results.

Data Aquisition
There are two types of data used in this project. The first one comes from the publicly available Matterport 3D tour scans.
We have prepared an internal tool exporting the equirectangular panoramas from each of the tours. We called it PanoramaTaker.
The result of its execution is a folder with equirectangular panoramas and one .json with details about their cooirdinates and metadata.
The second datasource is Ricoh Theta camera and multiple apartments we went through in a picture-by-picture manner and collected hundreds of images.

This datasets are then loaded to a hosted internal instance of the CVAT annotation tool.
For the classification task the datasets for each of the buildings is exported and manually parsed together because of the buggy functionality of CVAT's export.
At the moment of writing datastructures with annotations for a specific job are correct, but for a whole project are not. The formatting won't be recognized by 
neither pytorch nor tensorflow dataset processors. We manually analyze the results and prepare proper datasets. 
