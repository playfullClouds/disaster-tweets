import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import BertTokenizer, TFBertForSequenceClassification

from src.logger import log
from dataclasses import dataclass
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    base_dir: str = os.path.join('artifacts', 'data_transformation')
    destination_dir: str = os.path.join('artifacts', 'model_trainer')
    train_data: str = os.path.join(base_dir, 'train_BERT.csv')
    test_data: str = os.path.join(base_dir, 'test_BERT.csv')
    val_data: str = os.path.join(base_dir, 'val_BERT.csv')
    best_model_path: str = os.path.join(destination_dir, 'best_model')
    max_seq_length: int = 128  # Adjust based on your data

def f1_metric(y_true, y_pred):
    return tf.py_function(lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'), (y_true, y_pred), tf.float32)

class ModelTrainer:
    def __init__(self):
        log.info("Initializing ModelTrainer")
        self.config = ModelTrainerConfig()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)

    def pad_sequences(self, sequences, maxlen):
        # Padding sequences to the same length
        return np.array([np.pad(seq, (0, max(0, maxlen - len(seq))), 'constant') for seq in sequences])

    def train_model(self):
        log.info("Model training starting...")
        train_df = self.load_data(self.config.train_data)
        val_df = self.load_data(self.config.val_data)
        maxlen = self.config.max_seq_length

        # Ensure correct shaping
        train_df['input_ids'] = self.pad_sequences(train_df['input_ids'], maxlen)
        train_df['attention_mask'] = self.pad_sequences(train_df['attention_mask'], maxlen)
        val_df['input_ids'] = self.pad_sequences(val_df['input_ids'], maxlen)
        val_df['attention_mask'] = self.pad_sequences(val_df['attention_mask'], maxlen)

        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices(({
            'input_ids': train_df['input_ids'],
            'attention_mask': train_df['attention_mask']
        }, train_df['target'])).shuffle(len(train_df)).batch(32)
        val_dataset = tf.data.Dataset.from_tensor_slices(({
            'input_ids': val_df['input_ids'],
            'attention_mask': val_df['attention_mask']
        }, val_df['target'])).batch(32)

        # Debugging: Check shapes of inputs right before model fitting
        for inputs, targets in train_dataset.take(1):
            print("Debug: input_ids shape:", inputs['input_ids'].shape)
            print("Debug: attention_mask shape:", inputs['attention_mask'].shape)

        try:
            self.model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            self.model.fit(train_dataset, validation_data=val_dataset, epochs=3)
            log.info("Model training completed successfully")
        except Exception as e:
            log.error("Error during model training: ", e)
            raise

    def load_data(self, path):
        df = pd.read_csv(path)
        df['input_ids'] = df['input_ids'].apply(eval)
        df['attention_mask'] = df['attention_mask'].apply(eval)
        return df
