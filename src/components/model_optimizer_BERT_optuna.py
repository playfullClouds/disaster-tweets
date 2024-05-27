import os
import sys
import optuna
import tensorflow as tf


from sklearn.metrics import classification_report, f1_score
from transformers import TFBertForSequenceClassification, BertTokenizer


from src.logger import log
from dataclasses import dataclass
from src.exception import CustomException



@dataclass
class ModelOptimizerConfig:
    train_data_path: str = os.path.join('artifacts', 'data_transformation', 'BERT_SMOTE', 'train_data.tfrecord')
    val_data_path: str = os.path.join('artifacts', 'data_transformation', 'BERT_SMOTE', 'val_data.tfrecord')
    test_data_path: str = os.path.join('artifacts', 'data_transformation', 'BERT_SMOTE', 'test_data.tfrecord')
    model_save_path: str = os.path.join('artifacts', 'model_optimizer', 'optuna_model')
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5
    optuna_trials: int = 10  
    
    
class ModelOptimizerOptuna:
    def __init__(self) -> None:
        log.info("Initializing ModelTrainer")
        self.config = ModelOptimizerConfig()

        try:
            os.makedirs(os.path.dirname(self.config.model_save_path), exist_ok=True)
            log.info(f"Created necessary directories")
        except Exception as e:
            log.error("Failed to set up the directories.")
            raise CustomException(e, sys)

        log.info("Setting up BERT model and tokenizer")
        try:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
            log.info("BERT model and tokenizer setup completed")
        except Exception as e:
            log.error("Failed to set up the BERT model and tokenizer.")
            raise CustomException(e, sys)


    def _parse_tfrecord(self, example_proto):
        log.info("Parsing TFRecord example")
        try:
            feature_description = {
                'input_ids': tf.io.FixedLenFeature([self.config.max_length], tf.int64),
                'attention_mask': tf.io.FixedLenFeature([self.config.max_length], tf.int64),
                'target': tf.io.FixedLenFeature([], tf.int64),
            }
            return tf.io.parse_single_example(example_proto, feature_description)
        except Exception as e:
            log.error("Failed to parse TFRecord example")
            raise CustomException(e, sys)


    def _load_dataset(self, filepath):
        log.info(f"Loading dataset from {filepath}")
        try:
            dataset = tf.data.TFRecordDataset(filepath)
            dataset = dataset.map(self._parse_tfrecord)
            dataset = dataset.map(lambda x: (
                {
                    'input_ids': tf.cast(x['input_ids'], tf.int32),
                    'attention_mask': tf.cast(x['attention_mask'], tf.int32)
                },
                tf.cast(x['target'], tf.int32)
            ))
            dataset = dataset.shuffle(1000).batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
            log.info(f"Loaded dataset from {filepath} successfully")
            return dataset
        except Exception as e:
            log.error(f"Failed to load dataset from {filepath}")
            raise CustomException(e, sys)


    def _compute_f1_score(self, y_true, y_pred):
        log.info("Computing F1 score")
        try:
            return f1_score(y_true, y_pred, average='weighted')
        except Exception as e:
            log.error("Failed to compute F1 score")
            raise CustomException(e, sys)

    
    def _objective(self, trial):
        log.info("Starting a new Optuna trial")
        try:
            # Suggest hyperparameters
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 5e-5)
            num_train_epochs = trial.suggest_int('num_train_epochs', 1, 5)
            per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [8, 16, 32])

            log.info(f"Trial hyperparameters: learning_rate={learning_rate}, num_train_epochs={num_train_epochs}, batch_size={per_device_train_batch_size}")

            # Load datasets
            train_dataset = self._load_dataset(self.config.train_data_path)
            val_dataset = self._load_dataset(self.config.val_data_path)

            # Compile the model with the suggested hyperparameters
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]
            )
            
            log.info("Model compiled successfully")
            
            # Train the model
            self.model.fit(train_dataset, validation_data=val_dataset, epochs=num_train_epochs, batch_size=per_device_train_batch_size)

            log.info("Model training completed for the current trial")

            # Evaluate the model
            y_true, y_pred = [], []
            for x_batch, y_batch in val_dataset:
                logits = self.model.predict(x_batch).logits
                y_pred.extend(tf.argmax(logits, axis=1).numpy())
                y_true.extend(y_batch.numpy())

            f1 = self._compute_f1_score(y_true, y_pred)
            log.info(f"Trial completed with F1 score: {f1}")
            
            return f1
        
        except Exception as e:
            log.error("Error occurred during the Optuna trial")
            raise CustomException(e, sys)
    
    
    def train(self) -> None:
        log.info("Starting model training process with Optuna")
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(self._objective, n_trials=self.config.optuna_trials)

            log.info(f"Best trial: {study.best_trial.value}")
            log.info(f"Best hyperparameters: {study.best_trial.params}")

            # Train with best hyperparameters
            best_params = study.best_trial.params
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate']),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]
            )

            train_dataset = self._load_dataset(self.config.train_data_path)
            val_dataset = self._load_dataset(self.config.val_data_path)
            self.model.fit(train_dataset, validation_data=val_dataset, epochs=best_params['num_train_epochs'], batch_size=best_params['per_device_train_batch_size'])

            log.info("Evaluating the model on the test set")
            test_dataset = self._load_dataset(self.config.test_data_path)
            results = self.model.evaluate(test_dataset)
            log.info(f"Test set results: {results}")

            log.info("Generating classification report")
            y_true, y_pred = [], []
            for batch in test_dataset:
                X_batch, y_batch = batch
                y_true.extend(y_batch.numpy())
                y_pred.extend(tf.argmax(self.model.predict(X_batch).logits, axis=1).numpy())

            report = classification_report(y_true, y_pred, target_names=['not_informative', 'informative'])
            log.info(f"Classification Report:\n{report}")

            log.info("Saving the model")
            self.model.save_pretrained(self.config.model_save_path)
            self.tokenizer.save_pretrained(self.config.model_save_path)
            log.info(f"Model saved to {self.config.model_save_path}")

            log.info("Model training process successful.")

        except Exception as e:
            log.error("Error occurred during model training process")
            raise CustomException(e, sys)
