from src.ML_Project_Hardika.logger import logging
import sys
from src.ML_Project_Hardika.exception import CustomException
from src.ML_Project_Hardika.components.data_ingestion import DataIngestion
from src.ML_Project_Hardika.components.data_ingestion import DataIngestionConfig
from src.ML_Project_Hardika.components.data_transformation import DataTransformation
from src.ML_Project_Hardika.components.model_training import ModelTrainer


if __name__ == "__main__":
    logging.info("Starting the application...")

    try:
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed successfully.")

        data_transformation = DataTransformation()
        X_train, X_test, y_train, y_test, preprocessor_path = data_transformation.initiate_data_transformation(
            train_path, test_path
        )
        logging.info("Data transformation completed successfully.")

        model_trainer = ModelTrainer()
        final_accuracy = model_trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)
        logging.info(f"Model training completed successfully. Final accuracy: {final_accuracy}")
        print(f"Final accuracy: {final_accuracy}")

    except Exception as e:
        logging.info("Custom Exception occurred")
        raise CustomException(e, sys)