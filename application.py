from src.ML_Project_Hardika.logger import logging
import sys
from src.ML_Project_Hardika.exception import CustomException

if __name__ == "__main__":
    logging.info("Starting the application...")

    try:
        x = 1 / 0  # This will raise a ZeroDivisionError
    except Exception as e:
        logging.info("Custom Exception occurred")
        raise CustomException(e, sys)
    