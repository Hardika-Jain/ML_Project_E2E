from src.ML_Project_Hardika.logger import logging
import sys
from src.ML_Project_Hardika.exception import CustomException
from src.ML_Project_Hardika.components.data_ingestion import DataIngestion
from src.ML_Project_Hardika.components.data_ingestion import DataIngestionConfig

if __name__ == "__main__":
    logging.info("Starting the application...")

    try:
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed successfully.")
       
    except Exception as e:
        logging.info("Custom Exception occurred")
        raise CustomException(e, sys)
    