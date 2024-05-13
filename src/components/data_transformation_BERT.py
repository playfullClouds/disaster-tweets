import os
import sys
import torch
import joblib
import pandas as pd

from transformers import BertTokenizer
from sklearn.model_selection import train_test_split


from src.logger import log
from dataclasses import dataclass, field
from src.exception import CustomException



@dataclass
class DataTransformationConfig:
    source_dir: str = os.path.join('artifacts', 'data_cleaner', 'cleaned_data_BERT.csv')
    destination_dir: str = os.path.join('artifacts', 'data_transformation')  # Define the base directory for transformed data
    train_data: str = field(default='train_BERT.csv')
    test_data: str = field(default='test_BERT.csv')
    val_data: str = field(default='val_BERT.csv')
    # vectorizer_path: str = os.path.join(destination_dir, 'tfidf_vectorizer.pkl')
    
    
class DataTransformer:
    def __init__(self) -> None:
        """Initialize DataTransformer and set up configuration."""
        log.info("Initializing DataTransformer")
        self.config = DataTransformationConfig()
        
        try:
            os.makedirs(self.config.destination_dir, exist_ok=True)
            log.info(f"Created directory {self.config.destination_dir}")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            log.info("DataTransformer initialized successfully")
        except Exception as e:
            log.error("Failed to initialize DataTransformer")
            raise CustomException(e, sys)
        
        
    def load_cleaned_data(self) -> pd.DataFrame:
        """Load the cleaned dataset."""
        log.info(f"Loading cleaned data from {self.config.source_dir}")
        try:
            df = pd.read_csv(self.config.source_dir)
            log.info(f"Cleaned data loaded successfully from {self.config.source_dir}")
            return df
        except Exception as e:
            log.error(f"Failed to load cleaned data from {self.config.source_dir}")
            raise CustomException(e, sys)
        
        
    def fit_transform_tokenizer(self, texts):
        """Tokenize and encode the text using BERT tokenizer."""
        log.info("Fitting and transforming the TF-IDF vectorizer")
        try:
            encoded_data = self.tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors="pt")
            log.info("Text tokenized and encoded successfully")
            return encoded_data
        except Exception as e:
            log.error("Error during text tokenization and encoding")
            raise CustomException(e, sys)
        
    def encode_labels(self, labels):
        """Map 'informative' and 'not_informative' labels to numeric values."""
        log.info("Encoding target labels")
        try:
            mapping = {'informative': 1, 'not_informative': 0}
            encoded_labels = labels.map(mapping)
            # Ensure labels are integers
            # encoded_labels = encoded_labels.astype(int)
            if encoded_labels.isnull().any():
                raise ValueError("Found null values after encoding target labels")
            log.info("Target labels encoded successfully")
            return encoded_labels
        except Exception as e:
            log.error("Error during target label encoding")
            raise CustomException(e, sys)

        
    def split_and_save_data(self, X, attention_masks, y):
        """Split the dataset into training, validation, and test sets, then save them."""
        log.info("Splitting data into train, validation, and test sets")
        try:
            train_X, val_test_X, train_mask, val_test_mask, train_y, val_test_y = train_test_split(X, attention_masks, y, test_size=0.4, random_state=42)
            val_X, test_X, val_mask, test_mask, val_y, test_y = train_test_split(val_test_X, val_test_mask, val_test_y, test_size=0.5, random_state=42)

            # Update paths to include the destination directory
            train_path = os.path.join(self.config.destination_dir, self.config.train_data)
            val_path = os.path.join(self.config.destination_dir, self.config.val_data)
            test_path = os.path.join(self.config.destination_dir, self.config.test_data)

            train_df = pd.DataFrame({'input_ids': train_X.tolist(), 'attention_mask': train_mask.tolist(), 'target': train_y.tolist()})
            val_df = pd.DataFrame({'input_ids': val_X.tolist(), 'attention_mask': val_mask.tolist(), 'target': val_y.tolist()})
            test_df = pd.DataFrame({'input_ids': test_X.tolist(), 'attention_mask': test_mask.tolist(), 'target': test_y.tolist()})
            
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            test_df.to_csv(test_path, index=False)

            log.info(f"Data successfully split and saved: train ({train_path}), validation ({val_path}), test ({test_path})")
        except Exception as e:
            log.error("Error during data splitting and saving")
            raise CustomException(e)



        
        
    def transform_data(self) -> None:
        """Complete the data transformation process."""
        log.info("Starting data transformation")
        try:
            # Load cleaned data
            df = self.load_cleaned_data()

            # Use `cleanText` for text data and `text_info` for target labels
            texts = df['cleanText']
            labels = df['text_info']

             # Tokenize and encode text data using BERT tokenizer
            encoded_data = self.fit_transform_tokenizer(texts)
            X = encoded_data['input_ids']
            attention_masks = encoded_data['attention_mask']
            
            # Convert labels to torch tensors
            y = self.encode_labels(labels)
            
            # Split and save data into train, validation, and test sets
            self.split_and_save_data(X, attention_masks, y)


            log.info("Data transformation process completed successfully")
        except Exception as e:
            log.error("Error occurred during data transformation process")
            raise CustomException(e)