import os
import sys
import tensorflow as tf



from sklearn.metrics import classification_report, f1_score
from transformers import TFBertForSequenceClassification, BertTokenizer


from src.logger import log
from dataclasses import dataclass
from src.exception import CustomException



@dataclass
class ModelTrainingConfig:
    train_data_path: str = os.path.join('artifacts', 'data_transformation', 'BERT_SMOTE', 'train_data.tfrecord')
    val_data_path: str = os.path.join('artifacts', 'data_transformation', 'BERT_SMOTE', 'val_data.tfrecord')
    test_data_path: str = os.path.join('artifacts', 'data_transformation', 'BERT_SMOTE', 'test_data.tfrecord')
    model_save_path: str = os.path.join('artifacts', 'model_trainer', 'bert_model')
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5
    
    
    
class ModelTrainerPlainBERT:
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


    def _compute_f1_score(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average='weighted')


    def train(self) -> None:
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

            # Define custom callback to calculate F1 score after each epoch
            class F1ScoreCallback(tf.keras.callbacks.Callback):
                def __init__(self, val_data, model_trainer):
                    self.val_data = val_data
                    self.model_trainer = model_trainer
                    self.f1_scores = []

                def on_epoch_end(self, epoch, logs=None):
                    y_true = []
                    y_pred = []
                    for x_batch, y_batch in self.val_data:
                        logits = self.model.predict(x_batch).logits
                        y_pred.extend(tf.argmax(logits, axis=1).numpy())
                        y_true.extend(y_batch.numpy())
                    f1 = self.model_trainer._compute_f1_score(y_true, y_pred)
                    self.f1_scores.append(f1)
                    print(f" - val_f1_score: {f1}")

            f1_callback = F1ScoreCallback(val_dataset, self)

            log.info("Training the model")
            history = self.model.fit(train_dataset, validation_data=val_dataset, epochs=self.config.epochs, callbacks=[f1_callback])

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