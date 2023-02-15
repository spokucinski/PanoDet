import os


def get_dataset_file_paths(datasets_folder_path: str, dataset_file_extension: str) -> []:
    datasets_to_test = []
    for root, dirs, files in os.walk(datasets_folder_path):
        for file in files:
            if file.endswith(dataset_file_extension):
                datasets_to_test.append(os.path.join(root, file))

    return datasets_to_test


def get_dataset_name(dataset_file_path: str) -> str:

    # Split the path to dataset with use of os separator, take the last separated element ('datasetname.yaml')
    # split it again with the dot and take the first part - name of the .yaml file ('datasetname')

    return dataset_file_path.split(os.sep)[-1].split('.')[0]
