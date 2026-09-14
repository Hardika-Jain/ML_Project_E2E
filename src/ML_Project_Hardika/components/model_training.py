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

from sklearn.metrics import accuracy_score

from src.ML_Project_Hardika.utils import save_object, evaluate_model
from src.ML_Project_Hardika.exception import CustomException
from src.ML_Project_Hardika.logger import logging


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    # path where the best-performing trained model will be saved as a pickle file


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

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
                    # Tuning searches for the C that best balances this trade-off for OUR data.
                    "C": [0.01, 0.1, 1, 10]
                },
                "Decision Tree": {
                    # max_depth = how many yes/no splits deep the tree can go.
                    # Shallow (3) = simple tree, may miss real patterns (underfit).
                    # Deep (None = unlimited) = tree can memorize training data (overfit).
                    "max_depth": [3, 5, 10, None],
                    # criterion = the math used to decide which feature to split on at
                    # each node. 'gini' and 'entropy' usually give similar results but
                    # not always identical — tuning checks which works better here.
                    "criterion": ["gini", "entropy"]
                },
                "Random Forest": {
                    # n_estimators = how many individual decision trees to build and
                    # average together. More trees = more stable predictions, but
                    # slower to train with diminishing returns past a point.
                    "n_estimators": [100, 200],
                    # max_depth = same idea as Decision Tree, but applied to EACH tree
                    # inside the forest — controls how complex each individual tree gets.
                    "max_depth": [5, 10, None]
                },
                "Gradient Boosting": {
                    # n_estimators = how many trees are added sequentially, where each
                    # new tree tries to correct the errors of the previous ones.
                    "n_estimators": [100, 200],
                    # learning_rate = how much each new tree's correction counts.
                    # Small (0.01) = slow, cautious learning, needs more trees to converge.
                    # Larger (0.1) = faster learning, but risks overshooting/overfitting.
                    "learning_rate": [0.01, 0.1]
                },
                "AdaBoost": {
                    # n_estimators = how many weak learners (simple trees) are combined
                    # in sequence, each one focusing more on the previous one's mistakes.
                    "n_estimators": [50, 100],
                    # learning_rate = how strongly each weak learner's vote is weighted
                    # in the final combined prediction.
                    "learning_rate": [0.01, 0.1, 1]
                },
                "K-Neighbors": {
                    # n_neighbors = how many nearest data points 'vote' on the
                    # prediction for a new point.
                    # Small (3) = very sensitive to local noise (can overfit).
                    # Large (9) = smoother, more generalized boundary (can underfit).
                    "n_neighbors": [3, 5, 7, 9]
                },
                "XGBoost": {
                    # same idea as Gradient Boosting above — XGBoost is an optimized,
                    # faster implementation of the same boosting concept.
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1]
                },
                "CatBoost": {
                    # depth = tree depth for each boosting round (CatBoost's version
                    # of max_depth) — same underfit/overfit trade-off as before.
                    "depth": [4, 6, 8],
                    # learning_rate = same concept as the boosting models above.
                    "learning_rate": [0.01, 0.1]
                },
            }

            # Step 3: call evaluate_model() — this is where GridSearchCV actually
            # RUNS the tuning described above: for every model, it trains and
            # validates every hyperparameter combination in `params`, picks the
            # best-scoring one, refits it on the full training set, and scores
            # it on the test set. Returns {model_name: test_accuracy}.
            model_report: dict = evaluate_model(
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
            # NOTE: after evaluate_model() runs, `models[best_model_name]` already
            # holds the TUNED version (its hyperparameters were set to gs.best_params_
            # inside evaluate_model, not the untuned defaults from Step 1)

            # Step 5: reject a run where even the best tuned model is weak
            if best_model_score < 0.6:
                raise CustomException("No good model found — best score was below 0.6", sys)

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            # Step 6: save only the best (tuned) model to disk
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Step 7: final check — recompute accuracy on the saved model
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