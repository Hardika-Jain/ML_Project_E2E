from src.ML_Project_Hardika.logger import logging
import sys
from src.ML_Project_Hardika.exception import CustomException
from src.ML_Project_Hardika.components.data_ingestion import DataIngestion
from src.ML_Project_Hardika.components.data_ingestion import DataIngestionConfig
from src.ML_Project_Hardika.components.data_transformation import DataTransformation

from src.ML_Project_Hardika.logger import logging
import sys
from src.ML_Project_Hardika.exception import CustomException
from src.ML_Project_Hardika.components.data_ingestion import DataIngestion
from src.ML_Project_Hardika.components.data_ingestion import DataIngestionConfig
from src.ML_Project_Hardika.components.data_transformation import DataTransformation


if __name__ == "__main__":
    logging.info("Starting the application...")

    try:
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        # capturing the returned paths instead of discarding them
        logging.info("Data ingestion completed successfully.")

        data_transformation = DataTransformation()
        X_train, X_test, y_train, y_test, preprocessor_path = data_transformation.initiate_data_transformation(
            train_path, test_path
        )
        # passing those paths into transformation, which was imported but never used
        logging.info("Data transformation completed successfully.")

    except Exception as e:
        logging.info("Custom Exception occurred")
        raise CustomException(e, sys)

