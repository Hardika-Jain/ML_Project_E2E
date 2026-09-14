"""
1. What does "training" actually mean here?

Every one of these models (Logistic Regression, Random Forest, XGBoost, etc.) is just a mathematical function 
with adjustable internal numbers (called parameters or weights). "Training" means: feed the model your X_train
(features: price_usd_scaled, skin_tone_encoded, skin_type one-hot columns, brand_freq_encoded) alongside y_train 
(the true answer: was this recommended, 0 or 1), and let an algorithm adjust those internal numbers so the model's 
predictions get as close as possible to the true y_train values.

for something like Logistic Regression: it's learning a formula like

probability(recommended) = f(w1*price + w2*skin_tone + w3*skin_type_dry + w4*brand_freq + ...)
Training is the process of finding the best w1, w2, w3, w4... values — the ones that make its predictions 
on X_train match y_train as closely as possible. For tree-based models (Random Forest, XGBoost), "training" 
instead means learning a series of if/else split rules (e.g. "if price_usd_scaled > 0.5 and skin_type_dry = 1, 
predict recommended") that best separate recommended vs. not-recommended rows.
"""

import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse

import dagshub
import mlflow
import mlflow.sklearn

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from src.ML_Project_Hardika.utils import save_object, evaluate_model
from src.ML_Project_Hardika.exception import CustomException
from src.ML_Project_Hardika.logger import logging

# NEW: connects this script to your DagsHub repo's MLflow tracking server.
# First time this runs, it may open a browser tab asking you to log in to
# DagsHub and authorize — after that, it's remembered automatically.
dagshub.init(repo_owner='Hardika-Jain', repo_name='ML_Project_E2E', mlflow=True)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    # path where the best-performing trained model will be saved as a pickle file


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, y_test, y_pred, y_pred_proba):
        """
        NEW: computes every classification metric we care about in one place,
        called once for the best model right before logging to MLflow.
        """
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        return accuracy, precision, recall, f1, roc_auc

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        logging.info("Entered the model training component")

        try:
            # Step 1: candidate classifiers to try
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "AdaBoost": AdaBoostClassifier(random_state=42),
                "K-Neighbors": KNeighborsClassifier(),
                "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
                "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
            }

            # Step 2: hyperparameter grids — one per model, keys must match `models` above.
            # Every value here is a HYPERPARAMETER: a setting chosen BEFORE training,
            # not something the model learns from data on its own. GridSearchCV (called
            # in Step 3) will try every combination listed for each model and keep
            # whichever combination scores best on validation folds.
            params = {
                "Logistic Regression": {
                    # C = inverse regularization strength.
                    # Small C (0.01) = strong regularization = simpler decision boundary,
                    # less likely to overfit but may underfit.
                    # Large C (10) = weak regularization = fits training data more closely,
                    # risking overfitting if the data is noisy.
                    "C": [0.01, 0.1, 1, 10]
                },
                "Decision Tree": {
                    # max_depth = how many yes/no splits deep the tree can go.
                    "max_depth": [3, 5, 10, None],
                    # criterion = the math used to decide which feature to split on
                    "criterion": ["gini", "entropy"]
                },
                "Random Forest": {
                    # n_estimators = how many individual decision trees to build and average
                    "n_estimators": [100, 200],
                    # max_depth = same idea as Decision Tree, applied to EACH tree in the forest
                    "max_depth": [5, 10, None]
                },
                "Gradient Boosting": {
                    # n_estimators = how many trees are added sequentially
                    "n_estimators": [100, 200],
                    # learning_rate = how much each new tree's correction counts
                    "learning_rate": [0.01, 0.1]
                },
                "AdaBoost": {
                    # n_estimators = how many weak learners are combined in sequence
                    "n_estimators": [50, 100],
                    # learning_rate = how strongly each weak learner's vote is weighted
                    "learning_rate": [0.01, 0.1, 1]
                },
                "K-Neighbors": {
                    # n_neighbors = how many nearest data points 'vote' on the prediction
                    "n_neighbors": [3, 5, 7, 9]
                },
                "XGBoost": {
                    # same idea as Gradient Boosting — optimized, faster implementation
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1]
                },
                "CatBoost": {
                    # depth = tree depth for each boosting round
                    "depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.1]
                },
            }

            # Step 3: call evaluate_model() — runs GridSearchCV tuning for every model.
            # NEW: now returns TWO dicts — {model_name: test_score} AND
            # {model_name: actual_best_hyperparameters} — needed so we can log the
            # REAL winning hyperparameters to MLflow, not the full search grid.
            model_report: dict
            best_params_report: dict
            model_report, best_params_report = evaluate_model(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=params
            )
            logging.info(f"Model report: {model_report}")
            print("\n=== All Model Scores ===")
            for name, score in sorted(model_report.items(), key=lambda x: x[1], reverse=True):
                print(f"{name}: {score:.4f}")
            print()

            # Step 4: pick the model with the highest test score after tuning
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            best_params = best_params_report[best_model_name]
            # NOTE: best_model already holds the TUNED version (hyperparameters set
            # to gs.best_params_ inside evaluate_model, not the untuned defaults)

            print("This is the best model:")
            print(best_model_name)

            # Step 5: reject a run where even the best tuned model is weak
            if best_model_score < 0.6:
                raise CustomException("No good model found — best score was below 0.6", sys)

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            # Step 6 (NEW): point MLflow at the DagsHub Model Registry for this repo
            mlflow.set_registry_uri("https://dagshub.com/Hardika-Jain/ML_Project_E2E.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Step 7 (NEW): log ONLY the best model as one MLflow run — its real
            # tuned hyperparameters, all 5 classification metrics, and the model
            # artifact itself, registered under its own name on DagsHub
            with mlflow.start_run(run_name=best_model_name):
                y_pred = best_model.predict(X_test)
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]

                accuracy, precision, recall, f1, roc_auc = self.eval_metrics(y_test, y_pred, y_pred_proba)

                mlflow.log_params(best_params)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("roc_auc", roc_auc)

                # Model registry doesn't work with local file-based tracking —
                # only register when using a real remote server (DagsHub, here)
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model_name)
                else:
                    mlflow.sklearn.log_model(best_model, "model")

            # Step 8: save the best (tuned) model locally too, as before
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Step 9: final check — recompute accuracy on the saved model
            predicted = best_model.predict(X_test)
            final_accuracy = accuracy_score(y_test, predicted)

            return final_accuracy

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.ML_Project_Hardika.components.data_ingestion import DataIngestion
    from src.ML_Project_Hardika.components.data_transformation import DataTransformation

    ingestion_obj = DataIngestion()
    train_path, test_path = ingestion_obj.initiate_data_ingestion()

    transformation_obj = DataTransformation()
    X_train, X_test, y_train, y_test, preprocessor_path = transformation_obj.initiate_data_transformation(
        train_path, test_path
    )

    trainer_obj = ModelTrainer()
    print(trainer_obj.initiate_model_trainer(X_train, X_test, y_train, y_test))