# This is the data transformation component which cleans the train/test splits,
# then scales and encodes the features so they're ready for model training.

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.ML_Project_Hardika.logger import logging
from src.ML_Project_Hardika.exception import CustomException
from src.ML_Project_Hardika.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    # path where the dict of fitted transformers will be saved as a pickle file,
    # so predict_pipeline.py can load and reuse them later on new data


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        # creates an instance of the config class so we can access the path above via self.data_transformation_config

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Called from Step 2below. Takes a df already in memory and returns a
        cleaned version — it never reads or writes files itself.
        """
        try:
            # 4a. Keep only the columns this problem actually needs — drops
            #     everything else (review text, product name, demographics, etc.)
            columns_to_keep = [
                'product_id',
                'brand_name_product',
                'price_usd_product',
                'skin_type',
                'skin_tone',
                'is_recommended'
            ]
            df = df[[c for c in columns_to_keep if c in df.columns]]
            
#Handeling missing values and duplicates
            # 4b. Drop rows with a missing label — is_recommended is what we're
            #     predicting, so we can't guess it or impute it
            df = df.dropna(subset=['is_recommended'])

            # 4c. Drop rows with no skin_type — the whole problem statement is
            #     about skin-type suitability, so these rows are useless to us
            df = df.dropna(subset=['skin_type'])

            # 4d. skin_tone is a secondary feature, not the main one — so instead
            #     of losing rows, just fill missing values with a placeholder
            df['skin_tone'] = df['skin_tone'].fillna('NaN')

            # 4e. Drop fully identical duplicate rows
            df = df.drop_duplicates()

            # 4f. Drop more targeted duplicates — same product_id + same
            #     recommendation value, which is likely the same entry twice
            df = df.drop_duplicates(subset=['product_id', 'is_recommended'], keep='first')

            # 4g. Normalize brand name text — strip whitespace, lowercase,
            #     so "Laneige" and "laneige " aren't treated as different brands
            df['brand_name_product'] = df['brand_name_product'].str.strip().str.lower()

            # 4h. Collapse different "no answer" labels for skin_tone into one
            #     single 'unknown' category, so the encoder doesn't treat them separately
            df['skin_tone'] = df['skin_tone'].replace({'NaN': 'unknown', 'notSureST': 'unknown'})

            return df
        except Exception as e:
            raise CustomException(e, sys)
            # wraps any error above so we know exactly where it happened
# Step 3: read the train/test CSVs (paths came from Step 1) into memory as dataframes

    def initiate_data_transformation(self, train_path: str, test_path: str):
        logging.info("Entered the data transformation component")
        # logs that this step has started when running the pipeline

        try:
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            # this is the ONLY place this file reads data from; the raw Kaggle CSVs
            # were already read, merged, and split inside data_ingestion.py
            logging.info("Read train and test data for transformation")

# Step 4: clean both dataframes using the clean_data() helper method above

            train_df = self.clean_data(train_df)
            test_df = self.clean_data(test_df)
            # applies the exact same cleaning steps to both sets independently
            # (safe here since cleaning is just filtering/dropping, not fitting anything)
            logging.info("Completed data cleaning on train and test sets")

            # Step 5: split each cleaned dataframe into features (X) and label (y)
            X_train = train_df.drop(columns=['is_recommended'])
            y_train = train_df['is_recommended']

            X_test = test_df.drop(columns=['is_recommended'])
            y_test = test_df['is_recommended']

            # Step 6: transform price — log-transform (both sets), then scale (fit on train only)
            X_train['price_usd_log'] = np.log1p(X_train['price_usd_product'])
            X_test['price_usd_log'] = np.log1p(X_test['price_usd_product'])
            # log-transform first since price is heavily right-skewed;
            # this is applied to both sets the same way, it's not "fit" on anything

            price_scaler = StandardScaler()
            X_train['price_usd_scaled'] = price_scaler.fit_transform(X_train[['price_usd_log']])
            # FIT the scaler on train only — this calculates mean/SD from train data
            X_test['price_usd_scaled'] = price_scaler.transform(X_test[['price_usd_log']])
            # TRANSFORM test using the mean/SD learned from train — prevents data leakage

            # Step 7: encode skin_tone — ordinal, since it has a lightness ranking
            tone_order = [['unknown', 'porcelain', 'fair', 'fairLight', 'light', 'lightMedium',
                           'medium', 'mediumTan', 'tan', 'olive', 'rich', 'deep', 'dark', 'ebony']]
            tone_encoder = OrdinalEncoder(categories=tone_order, handle_unknown='use_encoded_value', unknown_value=-1)
            X_train['skin_tone_encoded'] = tone_encoder.fit_transform(X_train[['skin_tone']])
            # FIT on train — learns the category order and any unseen-value handling
            X_test['skin_tone_encoded'] = tone_encoder.transform(X_test[['skin_tone']])
            # TRANSFORM only on test

            # Step 8: encode skin_type — one-hot, since it's nominal with no natural order
            skin_type_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
            train_ohe = skin_type_encoder.fit_transform(X_train[['skin_type']])
            # FIT on train — learns which categories exist and creates the dummy columns
            test_ohe = skin_type_encoder.transform(X_test[['skin_type']])
            # TRANSFORM only on test; any skin_type test has that train never saw is ignored

            ohe_cols = skin_type_encoder.get_feature_names_out(['skin_type'])
            X_train[ohe_cols] = train_ohe
            X_test[ohe_cols] = test_ohe
            # attaching the new one-hot columns back onto the original dataframes

            # Step 9: encode brand_name_product — frequency encoding (fit on train only)
            brand_freq = X_train['brand_name_product'].value_counts(normalize=True)
            # calculates how often each brand appears, FROM TRAIN ONLY

            X_train['brand_freq_encoded'] = X_train['brand_name_product'].map(brand_freq)
            X_test['brand_freq_encoded'] = X_test['brand_name_product'].map(brand_freq).fillna(0)
            # any brand that appears in test but never in train is treated as "unknown/rare", frequency 0

            # Step 10: drop the original raw columns now that they're encoded, plus product_id
            drop_cols = ['skin_tone', 'skin_type', 'brand_name_product', 'price_usd_product', 'price_usd_log']
            X_train = X_train.drop(columns=drop_cols)
            X_test = X_test.drop(columns=drop_cols)

            # product_id is useful for tracing rows back to a product, but isn't a model feature
            X_train_model = X_train.drop(columns=['product_id'])
            X_test_model = X_test.drop(columns=['product_id'])

            logging.info(f"Final train shape: {X_train_model.shape}, test shape: {X_test_model.shape}")

            # Step 11: bundle all fitted transformers into one dict and save as a pickle,
            # so predict_pipeline.py can reload and reapply them later — this isn't a single
            # sklearn Pipeline/ColumnTransformer because the steps here (log+scale, manual
            # frequency map, mixed ordinal/one-hot) are fit separately
            preprocessing_artifacts = {
                'price_scaler': price_scaler,
                'tone_encoder': tone_encoder,
                'skin_type_encoder': skin_type_encoder,
                'brand_freq_map': brand_freq,
                'tone_order': tone_order,
                'model_columns': X_train_model.columns.tolist()
            }

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_artifacts
            )
            # saves the dict of fitted objects to artifacts/preprocessor.pkl
            logging.info("Saved preprocessing objects")

            # Step 12: return the transformed data + preprocessor path to whatever called this method
            return (
                X_train_model, X_test_model, y_train, y_test,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
            # if anything above fails, wrap the error in our CustomException and raise it


if __name__ == "__main__":
    from src.ML_Project_Hardika.components.data_ingestion import DataIngestion

    # Step 1: run ingestion — downloads/merges/splits raw data, returns train/test CSV paths
    ingestion_obj = DataIngestion()
    train_path, test_path = ingestion_obj.initiate_data_ingestion()

    # Step 2: run transformation on those paths — this is what kicks off Step 3 above,
    # this only executes when the file is run directly, not when it's imported elsewhere
    transformation_obj = DataTransformation()
    transformation_obj.initiate_data_transformation(train_path, test_path)