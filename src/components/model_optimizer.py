import os
import gc 
import sys
import joblib
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold


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
class ModelOptimizerConfig:
    base_dir: str = os.path.join('artifacts', 'data_transformation')
    destination_dir: str = os.path.join('artifacts', 'model_optimizer')
    train_data_dir: str = os.path.join(base_dir, 'train.csv')
    val_data_dir: str = os.path.join(base_dir, 'val.csv')
    test_data_dir: str = os.path.join(base_dir, 'test.csv')
    optimized_model_path: str = os.path.join(destination_dir, 'optimized_model.pkl')


class ModelOptimizer:
    def __init__(self) -> None:
        """Initialize ModelOptimizer and setup configuration"""
        log.info("Initializing ModelOptimizer")
        self.config = ModelOptimizerConfig()

        try:
            os.makedirs(self.config.destination_dir, exist_ok=True)
        except Exception as e:
            log.error("Failed to set up the model optimizer directories")
            raise CustomException(e, sys)


    def load_data(self, path: str):
        """Load the data from the base directory."""
        log.info(f"Loading data from from {path}")

        try:
            df = pd.read_csv(path)
            log.info(f"Data loaded successfully from {path}")
            X = df.drop('target', axis=1)
            y = df['target']
            return X, y
        except Exception as e:
            log.error(f"Failed to load data from {path}")
            raise CustomException(e, sys)


    def grid_search_and_ensemble(self) -> None:
        """Optimize models using Grid Search and then apply ensemble methods."""
        log.info("Model optimization and ensemble creation starting...")

        try:
            # Load training, validation, and test data
            X_train, y_train = self.load_data(self.config.train_data_dir)
            X_val, y_val = self.load_data(self.config.val_data_dir)
            X_test, y_test = self.load_data(self.config.test_data_dir)

            # Stratified K-Fold for cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            # Grid search parameter grids
            params = {
                'Logistic Regression': {
                    'logisticregression__C': [0.001, 0.1, 1, 10],
                    'logisticregression__penalty': ['l2'],
                    'logisticregression__max_iter': [500, 1000]
                    # 'logisticregression__solver': ['lbfgs', 'saga', 'liblinear']
                },
                'Random Forest': {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 20],
                    # 'criterion': ['gini', 'entropy'],
                    'min_samples_split': [2, 5]
                },
                'Gradient Boosting': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.1, 0.01],
                    'max_depth': [3, 5]
                },
                'XGBoost': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                },
                'CatBoost': {
                    'learning_rate': [0.01, 0.1],
                    'depth': [3, 5],
                    'iterations': [50, 100]
                }
                
            }

            # Models to optimize with grid search
            models = {
                'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                'Random Forest': RandomForestClassifier(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'XGBoost': XGBClassifier(eval_metric='mlogloss', use_label_encoder=False),
                'CatBoost': CatBoostClassifier(verbose=0),
            }

            # best_model = None
            # best_f1_test = 0
            # best_model_name = ""

            best_model = None
            best_f1_test = 0
            best_model_name = ""
            best_f1_gap = float('inf')

            # Initialize dictionary to hold optimized models
            optimized_models = {}

            # Perform grid search for each model
            for name, model in models.items():
                grid_params = params[name]
                log.info(f"Optimizing model with Grid Search: {name}")

                grid_search = GridSearchCV(
                    model,
                    grid_params,
                    cv=cv,
                    scoring=make_scorer(f1_score, average='weighted'),
                    n_jobs=1,
                    verbose=1
                )

                grid_search.fit(X_train, y_train)

                optimized_model = grid_search.best_estimator_
                optimized_models[name] = optimized_model

                # Perform cross-validation for all metrics
                cross_val_results = cross_validate(
                    optimized_model,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring={
                        'accuracy': make_scorer(accuracy_score),
                        'precision': make_scorer(precision_score, average='weighted'),
                        'recall': make_scorer(recall_score, average='weighted'),
                        'f1_weighted': make_scorer(f1_score, average='weighted')
                    }
                )

                # Log cross-validation scores
                log.info(f"{name} cross-validation scores:")
                log.info(f"  Accuracy: {cross_val_results['test_accuracy'].mean():.4f}")
                log.info(f"  Precision: {cross_val_results['test_precision'].mean():.4f}")
                log.info(f"  Recall: {cross_val_results['test_recall'].mean():.4f}")
                log.info(f"  F1-Score: {cross_val_results['test_f1_weighted'].mean():.4f}")

                log.info(f"{name} optimized parameters: {grid_search.best_params_}")
                log.info(f"{name} best cross-validation F1-score: {grid_search.best_score_:.4f}")

                # Evaluate the optimized model on training data
                y_train_pred = optimized_model.predict(X_train)
                accuracy_train = accuracy_score(y_train, y_train_pred)
                precision_train = precision_score(y_train, y_train_pred, average='weighted')
                recall_train = recall_score(y_train, y_train_pred, average='weighted')
                f1_train = f1_score(y_train, y_train_pred, average='weighted')

                log.info(f"{name} - Training Scores: Accuracy = {accuracy_train:.4f}, "
                         f"Precision = {precision_train:.4f}, Recall = {recall_train:.4f}, F1-Score = {f1_train:.4f}")

                # Evaluate the optimized model on validation data
                y_val_pred = optimized_model.predict(X_val)
                accuracy_val = accuracy_score(y_val, y_val_pred)
                precision_val = precision_score(y_val, y_val_pred, average='weighted')
                recall_val = recall_score(y_val, y_val_pred, average='weighted')
                f1_val = f1_score(y_val, y_val_pred, average='weighted')

                log.info(f"{name} - Validation Scores: Accuracy = {accuracy_val:.4f}, "
                         f"Precision = {precision_val:.4f}, Recall = {recall_val:.4f}, F1-Score = {f1_val:.4f}")

                # Evaluate the optimized model on test data
                y_test_pred = optimized_model.predict(X_test)
                accuracy_test = accuracy_score(y_test, y_test_pred)
                precision_test = precision_score(y_test, y_test_pred, average='weighted')
                recall_test = recall_score(y_test, y_test_pred, average='weighted')
                f1_test = f1_score(y_test, y_test_pred, average='weighted')

                log.info(f"{name} - Test Scores: Accuracy = {accuracy_test:.4f}, "
                         f"Precision = {precision_test:.4f}, Recall = {recall_test:.4f}, F1-Score = {f1_test:.4f}")


                f1_gap = abs(f1_train - f1_test)

                # Select the best model if the gap is < 6% and the test F1-score is higher than the best found so far
                if f1_gap < 0.06 and f1_test > best_f1_test:
                    best_f1_test = f1_test
                    best_f1_gap = f1_gap
                    best_model = optimized_model
                    best_model_name = name

            # Ensure that the best model is above the minimum F1 threshold
            if best_f1_test < 0.6:
                raise ValueError(f"Best model ({best_model_name}) has an F1-score below 0.6 ({best_f1_test:.4f})")

            if best_model is None:
                raise ValueError("No suitable model was found based on the F1-score gap criteria")


            # Create an ensemble model using Voting Classifier
            ensemble = VotingClassifier(
                estimators=[(name, model) for name, model in optimized_models.items()],
                voting='soft'
            )

            ensemble.fit(X_train, y_train)

            # Evaluate the ensemble model on the training data
            y_train_pred = ensemble.predict(X_train)
            accuracy_train = accuracy_score(y_train, y_train_pred)
            precision_train = precision_score(y_train, y_train_pred, average='weighted')
            recall_train = recall_score(y_train, y_train_pred, average='weighted')
            f1_train = f1_score(y_train, y_train_pred, average='weighted')

            log.info(f"Ensemble - Training Scores: Accuracy = {accuracy_train:.4f}, "
                     f"Precision = {precision_train:.4f}, Recall = {recall_train:.4f}, F1-Score = {f1_train:.4f}")

            # Evaluate the ensemble model on the validation set
            y_val_pred = ensemble.predict(X_val)
            accuracy_val = accuracy_score(y_val, y_val_pred)
            precision_val = precision_score(y_val, y_val_pred, average='weighted')
            recall_val = recall_score(y_val, y_val_pred, average='weighted')
            f1_val = f1_score(y_val, y_val_pred, average='weighted')

            log.info(f"Ensemble - Validation Scores: Accuracy = {accuracy_val:.4f}, "
                     f"Precision = {precision_val:.4f}, Recall = {recall_val:.4f}, F1-Score = {f1_val:.4f}")

            # Evaluate the ensemble model on the test data
            y_test_pred = ensemble.predict(X_test)
            accuracy_test = accuracy_score(y_test, y_test_pred)
            precision_test = precision_score(y_test, y_test_pred, average='weighted')
            recall_test = recall_score(y_test, y_test_pred, average='weighted')
            f1_test = f1_score(y_test, y_test_pred, average='weighted')

            log.info(f"Ensemble - Test Scores: Accuracy = {accuracy_test:.4f}, "
                     f"Precision = {precision_test:.4f}, Recall = {recall_test:.4f}, F1-Score = {f1_test:.4f}")

            
           # Calculate the ensemble training-test F1-score gap
            ensemble_f1_gap = abs(f1_train - f1_test)

            # Determine which model to save based on the ensemble vs. best individual model comparison
            if ensemble_f1_gap < 0.06 and f1_test > best_f1_test:
                joblib.dump(ensemble, self.config.optimized_model_path)
                log.info(f"Optimized ensemble model saved at {self.config.optimized_model_path}")
            else:
                joblib.dump(best_model, self.config.optimized_model_path)
                log.info(f"Best individual model ({best_model_name}) saved at {self.config.optimized_model_path}")
            

            log.info("Model optimization and ensemble creation completed successfully")

            # Memory Management - Delete unnecessary variables and perform garbage collection
            del X_train, y_train, X_val, y_val, X_test, y_test
            gc.collect()  # Force garbage collection

        except Exception as e:
            log.error("Error occurred during model optimization and ensemble creation")
            raise CustomException(e, sys)


