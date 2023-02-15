import os


def get_experiment_run_name(tested_model: str, image_size: int, epoch_size: int, batch_size: int) -> str:
    return f'{tested_model}_{image_size}_{epoch_size}_{batch_size}'


def get_done_experiments(results_path: str,
                         model_family: str,
                         tested_dataset_name: str,
                         realized_task: str) -> list[str]:
    results_path = __get_experiment_results_path(results_path,
                                                 model_family,
                                                 tested_dataset_name,
                                                 realized_task)

    __ensure_path_exists(results_path)
    return os.listdir(results_path)


def __get_experiment_results_path(results_path: str, model_family: str, tested_dataset_name: str, realized_task: str):
    return os.path.join(results_path, model_family, tested_dataset_name, realized_task)


def __ensure_path_exists(folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
