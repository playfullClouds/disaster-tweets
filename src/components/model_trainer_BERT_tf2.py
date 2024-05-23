import os
import sys
import tensorflow as tf

from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.metrics import classification_report


from src.logger import log
from dataclasses import dataclass, field
from src.exception import CustomException


@dataclass
class ModelTrainingConfig:
    train_data_path: str = os.path.join('artifacts', 'data_transformation', 'train_data.tfrecord')
    val_data_path: str = os.path.join('artifacts', 'data_transformation', 'val_data.tfrecord')
    test_data_path: str = os.path.join('artifacts', 'data_transformation', 'test_data.tfrecord')
    model_save_path: str = os.path.join('artifacts', 'model', 'bert_model')
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5

class ModelTrainer:
    def __init__(self) -> None:
        log.info("Initializing ModelTrainer")
        self.config = ModelTrainingConfig()
        
        try:
            os.makedirs(os.path.dirname(self.config.model_save_path), exist_ok=True)
            log.info(f"Created directory {os.path.dirname(self.config.model_save_path)}")
            log.info("Model directory setup completed")
        except Exception as e:
            log.error("Failed to set up the model directory.")
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
        feature_description = {
            'input_ids': tf.io.FixedLenFeature([self.config.max_length], tf.int64),
            'attention_mask': tf.io.FixedLenFeature([self.config.max_length], tf.int64),
            'target': tf.io.FixedLenFeature([], tf.int64),
        }
        return tf.io.parse_single_example(example_proto, feature_description)

    def _load_dataset(self, filepath):
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
        return dataset

    def train_model(self) -> None:
        log.info("Starting model training process")
        try:
            train_dataset = self._load_dataset(self.config.train_data_path)
            val_dataset = self._load_dataset(self.config.val_data_path)
            test_dataset = self._load_dataset(self.config.test_data_path)

            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            log.info("Model compilation completed")

            log.info("Training the model")
            history = self.model.fit(train_dataset, validation_data=val_dataset, epochs=self.config.epochs)

            log.info("Evaluating the model on the test set")
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
