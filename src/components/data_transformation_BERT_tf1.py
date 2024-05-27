import os
import sys
import pandas as pd
import tensorflow as tf

from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

from src.logger import log
from dataclasses import dataclass, field
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    source_dir: str = os.path.join('artifacts', 'data_cleaner', 'cleaned_data_BERT.csv')
    destination_dir: str = os.path.join('artifacts', 'data_transformation')
    output_file_train: str = os.path.join(destination_dir, 'train_data.tfrecord')
    output_file_val: str = os.path.join(destination_dir, 'val_data.tfrecord')
    output_file_test: str = os.path.join(destination_dir, 'test_data.tfrecord')
    max_length: int = 128  # Max length for BERT tokenizer

class DataTransformer:
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
    
    def transform(self, sentence):
        """Transforms a single sentence using BERT tokenizer."""
        log.info("Main transformation starting")
        try:
            # Tokenize and pad/truncate the sentence for BERT
            encoding = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=self.config.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='tf'
            )
            
            log.info("Main transformation successful.")
            return encoding['input_ids'][0].numpy(), encoding['attention_mask'][0].numpy()
            
        except Exception as e:
            log.error("Error in transforming the sentence.")
            raise CustomException(e, sys)
    
    def _serialize_example(self, input_ids, attention_mask, target):
        feature = {
            'input_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=input_ids)),
            'attention_mask': tf.train.Feature(int64_list=tf.train.Int64List(value=attention_mask)),
            'target': tf.train.Feature(int64_list=tf.train.Int64List(value=[target])),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
    
    def _write_tfrecord(self, dataset, filename):
        with tf.io.TFRecordWriter(filename) as writer:
            for input_ids, attention_mask, target in dataset:
                example = self._serialize_example(input_ids.numpy(), attention_mask.numpy(), target.numpy())
                writer.write(example)
        
    def transform_data(self) -> None:
        """Transforms an input file and writes the transformed data to output files."""
        log.info("Starting file transformation process")
        try:
            df = pd.read_csv(self.config.source_dir)
            log.info(f"Loaded data from {self.config.source_dir}")
            log.info(f"Initial DataFrame shape: {df.shape}")
            
            # Ensure required columns exist
            if 'cleanText' not in df.columns or 'text_info' not in df.columns:
                raise CustomException("Input file does not contain required columns 'cleanText' and 'text_info'", sys)
            
            # Apply the transformation function to the cleanText column
            df[['input_ids', 'attention_mask']] = df['cleanText'].apply(
                lambda x: pd.Series(self.transform(x))
            )
            
            # Convert text_info to binary labels
            df['target'] = df['text_info'].apply(lambda x: 1 if x == 'informative' else 0)
            
            # Keep only the necessary columns
            df = df[['input_ids', 'attention_mask', 'target']]
            
            log.info(f"Transformed DataFrame shape: {df.shape}")
            
            # Split data into train, validation, and test sets
            train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)  # 60% train, 40% temp
            val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)  # 20% val, 20% test
            
            log.info("Dataset Splitted")
            
            # Convert DataFrames to TensorFlow datasets
            def to_tf_dataset(dataframe):
                input_ids = tf.ragged.constant(dataframe['input_ids'].tolist(), dtype=tf.int32)
                attention_mask = tf.ragged.constant(dataframe['attention_mask'].tolist(), dtype=tf.int32)
                target = tf.constant(dataframe['target'].tolist(), dtype=tf.int32)
                
                dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_mask, target))
                return dataset

            train_dataset = to_tf_dataset(train_df)
            val_dataset = to_tf_dataset(val_df)
            test_dataset = to_tf_dataset(test_df)
            
            # Save TensorFlow datasets as TFRecord files
            self._write_tfrecord(train_dataset, self.config.output_file_train)
            self._write_tfrecord(val_dataset, self.config.output_file_val)
            self._write_tfrecord(test_dataset, self.config.output_file_test)
            
            log.info(f"Transformed data saved to {self.config.output_file_train}, {self.config.output_file_val}, {self.config.output_file_test}")
            log.info("File transformation process successful.")
                    
        except Exception as e:
            log.error(f"Error occurred while transforming data from {self.config.source_dir}")
            raise CustomException(e, sys)


