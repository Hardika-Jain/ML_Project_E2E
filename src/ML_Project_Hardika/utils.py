import os
import sys
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from src.ML_Project_Hardika.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models: dict, param: dict) -> dict:
    """
    Called from model_trainer.py Step 3. This is where hyperparameter tuning
    actually executes for every model.
    """
    try:
        report = {}

        for model_name, model in models.items():
            # look up this model's hyperparameter grid, defined in model_trainer.py
            para = param[model_name]

            # GridSearchCV: tries EVERY combination of hyperparameters in `para`.
            # For each combination, it splits X_train into cv=3 folds, trains on
            # 2 folds and validates on the 3rd, rotating 3 times, then averages
            # the score — this is what makes the tuning result reliable rather
            # than lucky on one split. n_jobs=-1 just runs these fits in parallel
            # across all CPU cores to speed this up.
            gs = GridSearchCV(model, para, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)
            # after this line, gs.best_params_ holds whichever hyperparameter
            # combination scored highest across the 3 validation folds

            # apply the winning hyperparameters back onto the actual model object,
            # then refit it on the FULL training set (not just 2/3 folds this time)
            # so it learns from all available training data using the tuned settings
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # now evaluate this tuned, fully-trained model on the untouched test set
            y_test_pred = model.predict(X_test)
            test_model_score = accuracy_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)