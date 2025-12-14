import os
import shutil
import kagglehub


def load_kaggle_data():
    target_dir = "data"

    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        return

    path = kagglehub.dataset_download("pdavpoojan/the-rvlcdip-dataset-test")

    shutil.copytree(path, target_dir, dirs_exist_ok=True)
    # Download latest version

    print("Path to dataset files:", path)


if __name__ == "__main__":
    load_kaggle_data()
    print("Kaggle data loaded.")
