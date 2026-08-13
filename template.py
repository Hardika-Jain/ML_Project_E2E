'''this will define the structure of the project and will create the necessary folders and files for the project'''

import os
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

project_name = "ML_Project_Hardika"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py", #creating a package for components folder which will have files for data preprocessing,etc
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_training.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/implementation/__init__.py",
    f"src/{project_name}/implementation/training_pipeline.py",
    f"src/{project_name}/implementation/prediction_pipeline.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/utils.py",
    f"src/{project_name}/logger.py",
    "application.py",
    "dockerfile"

]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
