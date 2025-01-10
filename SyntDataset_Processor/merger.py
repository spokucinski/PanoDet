# Script used to merge multiple sub-parts of datasets into a single, bigger one

data_sources = os.listdir(DATA_PATH)
data_sources = [datasource for datasource in data_sources if os.path.isdir(os.path.join(DATA_PATH, datasource))]
for data_source in data_sources:
    subset_images_path = os.path.join(DATA_PATH, data_source, "images")
    subset_ann_path = os.path.join(DATA_PATH, data_source, "annotations.xml")
    dataset.merge_dir(data_path=subset_images_path, labels_path=subset_ann_path, dataset_type=IMPORT_TYPE)