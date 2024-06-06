import os
import gc
import sys
import joblib
import pandas as pd


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate

# from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    make_scorer
)


from src.logger import log
from dataclasses import dataclass
from src.exception import CustomException


@dataclass
class ModelTrainerConfig:
    base_dir: str = os.path.join('artifacts', 'data_transformation')
    destination_dir: str = os.path.join('artifacts', 'model_trainer')
    train_data: str = os.path.join(base_dir, 'train_SMOTE.csv')
    test_data: str = os.path.join(base_dir, 'test_SMOTE.csv')
    val_data: str = os.path.join(base_dir, 'val_SMOTE.csv')
    best_model_path: str = os.path.join(destination_dir, 'best_model_SMOTE.pkl')
    
    
class ModelTrainer:
    def __init__(self) -> None:
        """Initialize ModelTrainer and setup configurations."""
        log.info("Initializing ModelTrainer")
        self.config = ModelTrainerConfig()

        try:
            os.makedirs(self.config.destination_dir, exist_ok=True)
        except Exception as e:
            log.error(f"Error setting up the model trainer")
            raise CustomException(e, sys)

    def load_data(self, path: str):
        """Load the dataset from the specified path"""
        log.info(f"Loading data from {path}")

        try:
            df = pd.read_csv(path)
            log.info(f"Data loaded successfully from {path}")
            X = df.drop('target', axis=1)
            y = df['target']
            return X, y
        except Exception as e:
            log.error(f"Failed to load data from {path}")
            raise CustomException(e, sys)

    def cross_validation_and_test(self) -> None:
        """Train multiple models using cross-validation, select the best model based on multiple metrics, and test it."""
        log.info("Model training and evaluation starting...")

        try:
            
            # Check if the best model file already exists
            if os.path.exists(self.config.best_model_path):
                log.info("Best model already exists. Skipping model training.")
                return
            
            # Load training, validation, and test data
            X_train, y_train = self.load_data(self.config.train_data)
            X_val, y_val = self.load_data(self.config.val_data)
            X_test, y_test = self.load_data(self.config.test_data)

            # Define the classifiers to compare
            models = {
                'Logistic Regression': LogisticRegression(max_iter=500),
                'Random Forest': RandomForestClassifier(n_estimators=100),
                'Gradient Boosting': GradientBoostingClassifier(),
                'Naive Bayes': MultinomialNB(),
                'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
                'XGBoost': XGBClassifier(eval_metric='mlogloss', use_label_encoder=False),
                'CatBoost': CatBoostClassifier(verbose=0)
            }

            best_model = None
            smallest_f1_gap = float('inf')
            best_model_name = ""

            # Use F1-score as the scoring metric for cross-validation
            scoring_metrics = {
                'accuracy': make_scorer(accuracy_score),
                'precision_weighted': make_scorer(precision_score, average='weighted'),
                'recall_weighted': make_scorer(recall_score, average='weighted'),
                'f1_weighted': make_scorer(f1_score, average='weighted')
            }

            # Train and evaluate each model using cross-validation
            for name, model in models.items():
                log.info(f"Training model with cross-validation: {name}")

                # Perform cross-validation with multiple metrics
                cv_results = cross_validate(model, X_train, y_train, cv=5, scoring=scoring_metrics)

                # Log cross-validation scores
                log.info(f"{name} cross-validation scores:")
                log.info(f"  Accuracy: {cv_results['test_accuracy'].mean():.4f}")
                log.info(f"  Precision: {cv_results['test_precision_weighted'].mean():.4f}")
                log.info(f"  Recall: {cv_results['test_recall_weighted'].mean():.4f}")
                log.info(f"  F1-Score: {cv_results['test_f1_weighted'].mean():.4f}")

                
                
                # Train the model on training data
                model.fit(X_train, y_train)


                # Log training data metrics
                y_train_pred = model.predict(X_train)
                accuracy_train = accuracy_score(y_train, y_train_pred)
                precision_train = precision_score(y_train, y_train_pred, average='weighted')
                recall_train = recall_score(y_train, y_train_pred, average='weighted')
                f1_train = f1_score(y_train, y_train_pred, average='weighted')

                log.info(f"{name} - Training Scores: Accuracy = {accuracy_train:.4f}, "
                         f"Precision = {precision_train:.4f}, Recall = {recall_train:.4f}, F1-Score = {f1_train:.4f}")

                # Log validation data metrics
                y_val_pred = model.predict(X_val)
                accuracy_val = accuracy_score(y_val, y_val_pred)
                precision_val = precision_score(y_val, y_val_pred, average='weighted')
                recall_val = recall_score(y_val, y_val_pred, average='weighted')
                f1_val = f1_score(y_val, y_val_pred, average='weighted')

                log.info(f"{name} - Validation Scores: Accuracy = {accuracy_val:.4f}, "
                         f"Precision = {precision_val:.4f}, Recall = {recall_val:.4f}, F1-Score = {f1_val:.4f}")

                # Log test data metrics
                y_test_pred = model.predict(X_test)
                accuracy_test = accuracy_score(y_test, y_test_pred)
                precision_test = precision_score(y_test, y_test_pred, average='weighted')
                recall_test = recall_score(y_test, y_test_pred, average='weighted')
                f1_test = f1_score(y_test, y_test_pred, average='weighted')

                log.info(f"{name} - Test Scores: Accuracy = {accuracy_test:.4f}, "
                         f"Precision = {precision_test:.4f}, Recall = {recall_test:.4f}, F1-Score = {f1_test:.4f}")

                # Determine the F1-score gap between training and test data
                f1_gap = abs(f1_train - f1_test)

                # Select the model based on the smallest F1-score gap between training and test data
                if f1_gap <= 0.05 and f1_gap < smallest_f1_gap and f1_test >= 0.6:
                    smallest_f1_gap = f1_gap
                    best_f1_test = f1_test
                    best_model = model
                    best_model_name = name

            # Ensure that the best model is above the minimum F1 threshold
            if best_f1_test < 0.6:
                raise ValueError(f"Best model ({best_model_name}) has an F1-score below 0.6 ({best_f1_test:.4f})")


            if best_model is None:
                raise ValueError("No suitable model was found")

            # Train the best model on the training data
            best_model.fit(X_train, y_train)

            # Save the best model after evaluating it with the test data
            joblib.dump(best_model, self.config.best_model_path)
            log.info(f"Best model ({best_model_name}) saved at {self.config.best_model_path}")

            log.info("Model training and evaluation completed successfully")
        except Exception as e:
            log.error(f"Error during model training and evaluation")
            raise CustomException(e, sys)
        finally:
            # Release memory
            if 'X_train' in locals():
                del X_train
            if 'y_train' in locals():
                del y_train
            if 'X_val' in locals():
                del X_val
            if 'y_val' in locals():
                del y_val
            if 'X_test' in locals():
                del X_test
            if 'y_test' in locals():
                del y_test
            gc.collect()

        
        

# if __name__ == "__main__":
#     trainer = ModelTrainer()
#     trainer.cross_validation_and_test()
    
    
    
    
STAGE_NAME = "Model Training"
try:
   log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   trainer = ModelTrainer()
   trainer.cross_validation_and_test()
   log.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        log.exception(f"Exception occurred during {STAGE_NAME}")
        raise CustomException(e, sys)
    