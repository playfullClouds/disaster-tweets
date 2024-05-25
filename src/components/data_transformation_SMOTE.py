import os
import sys
import joblib
import pandas as pd


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


from src.logger import log
from dataclasses import dataclass, field
from src.exception import CustomException



@dataclass
class DataTransformationConfig:
    source_dir: str = os.path.join('artifacts', 'data_cleaner', 'cleaned_data.csv')
    destination_dir: str = os.path.join('artifacts', 'data_transformation') 
    train_data: str = field(default='train_SMOTE.csv')
    test_data: str = field(default='test_SMOTE.csv')
    val_data: str = field(default='val_SMOTE.csv')
    vectorizer_path: str = os.path.join(destination_dir, 'tfidf_vectorizer_SMOTE.pkl')
    
    

class DataTransformer:
    def __init__(self) -> None:
        """Initialize DataTransformer and set up configuration."""
        log.info("Initializing DataTransformer")
        self.config = DataTransformationConfig()
        
        try:
            os.makedirs(self.config.destination_dir, exist_ok=True)
            log.info(f"Created directory {self.config.destination_dir}")
            self.vectorizer = TfidfVectorizer(
                sublinear_tf=True,
                min_df=10,
                norm='l2',
                ngram_range=(1, 2),
                stop_words='english'
            )
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
        
        
    def fit_transform_vectorizer(self, texts):
        """Fit and transform the TF-IDF vectorizer."""
        log.info("Fitting and transforming the TF-IDF vectorizer")
        try:
            vectorized_data = self.vectorizer.fit_transform(texts)
            log.info("TF-IDF vectorizer fitted and transformed successfully")
            return vectorized_data
        except Exception as e:
            log.error("Error during TF_IDF vectorization")
            raise CustomException(e, sys)
        
        
    def transform_vectorizer(self, texts):
        """Transform the TF-IDF vectorizer."""
        log.info("Transforming the TF-IDF vectorizer")
        try:
            vectorized_data = self.vectorizer.transform(texts)
            log.info("TF-IDF vectorizer transformed successfully")
            return vectorized_data
        except Exception as e:
            log.error("Error during TF-IDF vectorization")
            raise CustomException(e, sys)
        
        
    def encode_labels(self, labels):
        """Map 'informative' and 'not_informative' labels to numeric values."""
        log.info("Encoding target labels")
        try:
            mapping = {'informative': 1, 'not_informative': 0}
            encoded_labels = labels.map(mapping)
            if encoded_labels.isnull().any():
                raise ValueError("Found null values after encoding target labels")
            log.info("Target labels encoded successfully")
            return encoded_labels
        except Exception as e:
            log.error("Error during target label encoding")
            raise CustomException(e, sys)
        
        
    def split_and_save_data(self, X, y):
        """Split the dataset into training, validation, and test sets, then save them."""
        log.info("Splitting data into train, validation, and test sets")
        try:
            train_set, temp_set, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
            val_set, test_set, y_val, y_test = train_test_split(temp_set, y_temp, test_size=0.5, random_state=42)

            # Transform the data using the vectorizer
            X_train = self.fit_transform_vectorizer(train_set['cleanText'])
            X_val = self.transform_vectorizer(val_set['cleanText'])
            X_test = self.transform_vectorizer(test_set['cleanText'])

            # Apply SMOTE to the vectorized training data
            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

            # Combine transformed text data and encoded labels into a single DataFrame
            train_transformed_df = pd.DataFrame(X_train_smote.toarray())
            train_transformed_df['target'] = y_train_smote
            val_transformed_df = pd.DataFrame(X_val.toarray())
            val_transformed_df['target'] = y_val.values
            test_transformed_df = pd.DataFrame(X_test.toarray())
            test_transformed_df['target'] = y_test.values

            # Save the datasets
            train_path = os.path.join(self.config.destination_dir, self.config.train_data)
            val_path = os.path.join(self.config.destination_dir, self.config.val_data)
            test_path = os.path.join(self.config.destination_dir, self.config.test_data)

            train_transformed_df.to_csv(train_path, index=False)
            val_transformed_df.to_csv(val_path, index=False)
            test_transformed_df.to_csv(test_path, index=False)

            log.info(f"Data successfully split and saved: train ({train_path}), validation ({val_path}), test ({test_path})")
        except Exception as e:
            log.error("Error during data splitting and saving")
            raise CustomException(e, sys)


    def save_vectorizer(self):
        """Save the TF-IDF vectorizer object to disk."""
        log.info(f"Saving TF-IDF vectorizer to {self.config.vectorizer_path}")
        try:
            joblib.dump(self.vectorizer, self.config.vectorizer_path)
            log.info("TF-IDF vectorizer saved successfully")
        except Exception as e:
            log.error(f"Error saving TF-IDF vectorizer to {self.config.vectorizer_path}")
            raise CustomException(e, sys)


    def transform_data(self) -> None:
        """Complete the data transformation process."""
        log.info("Starting data transformation")
        try:
            # Load cleaned data
            df = self.load_cleaned_data()

            # Use `cleanText` for text data and `text_info` for target labels
            X = df[['cleanText']]
            labels = df['text_info']

            # Encode labels
            y = self.encode_labels(labels)

            # Split and save data into train, validation, and test sets
            self.split_and_save_data(X, y)

            # Save the vectorizer for later use
            self.save_vectorizer()

            log.info("Data transformation process completed successfully")
        except Exception as e:
            log.error("Error occurred during data transformation process")
            raise CustomException(e, sys)