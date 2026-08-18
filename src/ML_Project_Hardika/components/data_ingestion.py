#This is the data ingestion component which will read the data from the csv file and split it into train and test datasets.
import os
import sys
from src.ML_Project_Hardika.logger import logging
from src.ML_Project_Hardika.exception import CustomException
import pandas as pd
import glob
from sklearn.model_selection import GroupShuffleSplit

from dataclasses import dataclass


# creating this class just stores file paths you read from and write to, so you can change them in one place if needed
@dataclass
class DataIngestionConfig:

    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
    # path where the full merged dataset (before splitting) will be saved, in the artifacts folder I created

    train_data_path: str = os.path.join('artifacts', 'train.csv')
    # path where the training split will be saved in the artifacts folder I created

    test_data_path: str = os.path.join('artifacts', 'test.csv')
    # path where the testing split will be saved in the artifacts folder I created


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        # creates an instance of the config class so we can access the paths above via self.ingestion_config

        self.folder = "/Users/hardikajain/Downloads/ML_project_dataset"
        # folder on disk where your raw CSV files live

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        # writes a log entry so we know this step started when running the pipeline

        try:
            #code to read the data from the csv file and split it into train and test datasets
            # DATASET 1 - Sephora

            product_info = pd.read_csv(os.path.join(self.folder, "product_info.csv"))
            # reads the product_info.csv file into a DataFrame

            review_files = glob.glob(os.path.join(self.folder, "reviews_*.csv"))
            # finds every file in the folder whose name starts with "reviews_" (there are 5 of them)

            reviews = pd.concat([pd.read_csv(f) for f in review_files], ignore_index=True)
            # reads each of those 5 files into a DataFrame, then stacks them into one combined DataFrame
            # ignore_index=True re-numbers the rows so there are no duplicate index values across files

            logging.info("Read product_info and reviews CSV files")
            # confirms both datasets were loaded successfully

            merged_data = reviews.merge(
                product_info,
                on='product_id',
                how='inner',
                suffixes=('_review', '_product')
            )
            # joins reviews and product_info together using product_id as the common key
            # how='inner' keeps only rows where product_id exists in both files
            # suffixes renames overlapping column names (e.g. price_usd) so we know which file each came from

            logging.info(f"Merged dataset shape: {merged_data.shape}")
            # logs the row/column count of the merged dataset, useful for sanity-checking later

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            # creates the 'artifacts' folder if it doesn't already exist
            #all the data stroed in the dataframe will be stored in the train data path.
            # exist_ok=True means it won't throw an error if the folder is already there

            merged_data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            # saves the full merged dataset to artifacts/raw.csv, before any splitting happens
            # index=False means we don't write pandas' row numbers into the CSV

            logging.info("Train test split initiated")
            # logs that we're about to split the data

            gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
            # sets up the splitter: 20% of the data goes to test, 1 split is generated
            # random_state=42 makes the split reproducible every time this runs
            # using GroupShuffleSplit instead of plain train_test_split because our data has
            # multiple reviews per product_id — grouping keeps all reviews of one product
            # entirely in either train or test, so we don't leak product info across the split

            train_idx, test_idx = next(gss.split(merged_data, groups=merged_data['product_id']))
            # generates the actual row indices for train and test
            # groups=merged_data['product_id'] ensures all reviews of the same product stay together

            train_set, test_set = merged_data.iloc[train_idx], merged_data.iloc[test_idx]
            # selects the rows belonging to the training set and test set using the indices from above

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            # saves the training set to artifacts/train.csv

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            # saves the test set to artifacts/test.csv

            logging.info("Ingestion of the data is completed")
            # logs that this step finished successfully

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            # returns the file paths so the next pipeline step (data transformation) knows where to find them

        except Exception as e:
            raise CustomException(e, sys)
            # if anything above fails, wrap the error in our CustomException and raise it
            # sys is passed in so CustomException can extract the line number/file where the error happened


if __name__ == "__main__":
    obj = DataIngestion()
    # creates an instance of the DataIngestion class

    obj.initiate_data_ingestion()
    # runs the ingestion process — this only executes when the file is run directly,
    # not when it's imported into another script (e.g. main.py)