import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf


from transformers import BertTokenizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


from src.logger import log
from dataclasses import dataclass
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    source_dir: str = os.path.join('artifacts', 'data_cleaner', 'cleaned_data_BERT.csv')
    destination_dir: str = os.path.join('artifacts', 'data_transformation', 'BERT_SMOTE')
    output_file_train: str = os.path.join(destination_dir, 'train_data.tfrecord')
    output_file_val: str = os.path.join(destination_dir, 'val_data.tfrecord')
    output_file_test: str = os.path.join(destination_dir, 'test_data.tfrecord')
    max_length: int = 128  # Max length for BERT tokenizer


class DataTransformerTfSMOTE:
    def __init__(self) -> None:
        log.info("Initializing DataTransformer")
        self.config = DataTransformationConfig()
        
        try:
            os.makedirs(self.config.destination_dir, exist_ok=True)
            log.info(f"Created directory {self.config.destination_dir}")
            log.info("Data Transformation directory setup completed")
        except Exception as e:
            log.error("Failed to set up the data transformation directories.")
            raise CustomException(e, sys)
        
        # Initialize BERT tokenizer
        log.info("Setting up BERT tokenizer for DataTransformer")
        try:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            log.info("BERT tokenizer setup completed")
        except Exception as e:
            log.error("Failed to set up the BERT tokenizer.")
            raise CustomException(e, sys)
    
    def load_data(self):
        try:
            df = pd.read_csv(self.config.source_dir)
            log.info(f"Loaded data from {self.config.source_dir}")
            log.info(f"Initial DataFrame shape: {df.shape}")
            
            # Ensure required columns exist
            if 'cleanText' not in df.columns or 'text_info' not in df.columns:
                raise CustomException("Input file does not contain required columns 'cleanText' and 'text_info'", sys)
            
            # Convert text_info to binary labels
            df['target'] = df['text_info'].apply(lambda x: 1 if x == 'informative' else 0)
            return df
        except Exception as e:
            log.error("Failed to load data.")
            raise CustomException(e, sys)

    def split_data(self, df):
        try:
            train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df['target'])  # 60% train, 40% temp
            val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['target'])  # 20% val, 20% test
            log.info("Dataset split into train, validation, and test sets")
            return train_df, val_df, test_df
        except Exception as e:
            log.error("Failed to split data.")
            raise CustomException(e, sys)

    def tokenize_sentences(self, sentences):
        try:
            encoding = self.tokenizer.batch_encode_plus(
                sentences.tolist(),
                add_special_tokens=True,
                max_length=self.config.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='np'
            )
            log.info("Sentences tokenized successfully")
            return encoding['input_ids'], encoding['attention_mask']
        except Exception as e:
            log.error("Failed to tokenize sentences.")
            raise CustomException(e, sys)

    def apply_smote(self, input_ids, attention_masks, targets):
        try:
            smote = SMOTE(random_state=42)
            log.info(f"Applying SMOTE: input_ids.shape = {input_ids.shape}, targets.shape = {targets.shape}")
            combined = np.hstack((input_ids, attention_masks))
            combined_resampled, targets_resampled = smote.fit_resample(combined, targets)
            input_ids_resampled = combined_resampled[:, :self.config.max_length]
            attention_masks_resampled = combined_resampled[:, self.config.max_length:]
            log.info("Applied SMOTE successfully")
            return input_ids_resampled, attention_masks_resampled, targets_resampled
        except Exception as e:
            log.error("Failed to apply SMOTE.")
            raise CustomException(e, sys)

    def transform_data(self, df, apply_smote=False):
        try:
            texts = df['cleanText'].values
            targets = df['target'].values
            input_ids, attention_masks = self.tokenize_sentences(texts)
            
            # Ensure input_ids and targets have the same length
            if len(input_ids) != len(targets):
                log.error(f"Inconsistent lengths before SMOTE: input_ids has {len(input_ids)} samples, targets has {len(targets)} samples")
                raise CustomException(f"Inconsistent lengths: input_ids has {len(input_ids)} samples, targets has {len(targets)} samples", sys)
            
            if apply_smote:
                input_ids, attention_masks, targets = self.apply_smote(input_ids, attention_masks, targets)
            
            log.info("Data transformation completed")
            return input_ids, attention_masks, targets
        except Exception as e:
            log.error("Failed to transform data.")
            raise CustomException(e, sys)

    def to_tf_dataset(self, input_ids, attention_masks, targets):
        try:
            dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_masks, targets))
            log.info("Converted data to TF dataset successfully")
            return dataset
        except Exception as e:
            log.error("Failed to convert data to TF dataset.")
            raise CustomException(e, sys)

    def _serialize_example(self, input_ids, attention_mask, target):
        try:
            feature = {
                'input_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=input_ids)),
                'attention_mask': tf.train.Feature(int64_list=tf.train.Int64List(value=attention_mask)),
                'target': tf.train.Feature(int64_list=tf.train.Int64List(value=[target])),
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            return example_proto.SerializeToString()
        except Exception as e:
            log.error("Failed to serialize example.")
            raise CustomException(e, sys)
    
    def _write_tfrecord(self, dataset, filename):
        try:
            with tf.io.TFRecordWriter(filename) as writer:
                for input_ids, attention_mask, target in dataset:
                    example = self._serialize_example(input_ids.numpy(), attention_mask.numpy(), target.numpy())
                    writer.write(example)
            log.info(f"TFRecord file written: {filename}")
        except Exception as e:
            log.error("Failed to write TFRecord file.")
            raise CustomException(e, sys)

    def save_tfrecords(self, train_dataset, val_dataset, test_dataset):
        try:
            self._write_tfrecord(train_dataset, self.config.output_file_train)
            self._write_tfrecord(val_dataset, self.config.output_file_val)
            self._write_tfrecord(test_dataset, self.config.output_file_test)
            log.info(f"Transformed data saved to {self.config.output_file_train}, {self.config.output_file_val}, {self.config.output_file_test}")
        except Exception as e:
            log.error("Failed to save TFRecord files.")
            raise CustomException(e, sys)

    def run_transformation(self):
        try:
            df = self.load_data()
            train_df, val_df, test_df = self.split_data(df)

            train_input_ids, train_attention_masks, train_targets = self.transform_data(train_df, apply_smote=True)
            val_input_ids, val_attention_masks, val_targets = self.transform_data(val_df)
            test_input_ids, test_attention_masks, test_targets = self.transform_data(test_df)

            train_dataset = self.to_tf_dataset(train_input_ids, train_attention_masks, train_targets)
            val_dataset = self.to_tf_dataset(val_input_ids, val_attention_masks, val_targets)
            test_dataset = self.to_tf_dataset(test_input_ids, test_attention_masks, test_targets)

            self.save_tfrecords(train_dataset, val_dataset, test_dataset)
        except Exception as e:
            log.error("Data transformation process failed.")
            raise CustomException(e, sys)