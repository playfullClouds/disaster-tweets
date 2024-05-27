import os
import re
import sys
import nltk
import pandas as pd


from nltk.corpus import stopwords
from transformers import BertTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


from src.logger import log
from dataclasses import dataclass, field
from src.exception import CustomException



@dataclass
class DataCleaningConfig:
    source_dir: str = os.path.join('artifacts', 'data_ingestion')
    destination_dir: str = os.path.join('artifacts', 'data_cleaner')
    input_file: str = 'tweets.csv'
    output_file: str = field(default='cleaned_data_BERT.csv')
    
    
class DataCleaner:
    def __init__(self) -> None:
        log.info("Initializing DataCleaner")
        self.config = DataCleaningConfig()
        
        try:
            
            os.makedirs(self.config.destination_dir, exist_ok=True)
            log.info(f"Created directory {self.config.destination_dir}")

            log.info("Data Cleaning directory setup completed")
        except Exception as e:
            log.error("Failed to set up the data cleaning directories.")
            raise CustomException(e, sys)
        
        # Initialize NLP tools and stopwords
        log.info("Setting up NLP tools for DataCleaner")
        try:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.stop_words = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
            log.info("NLP tools setup completed")
        except Exception as e:
            log.error("Failed to set up the NLP tools.")
            raise CustomException
        
    
    def clean(self, sentence):
        """Cleans a single sentence by removing unwanted tokens and applying normalization."""
        log.info("Main cleaning starting")
        try:
            
            # If the sentence is None or not a string, handle it gracefully
            if not sentence or not isinstance(sentence, str):
                log.warning(f"Invalid sentence value encountered: {sentence}")
                return ""  # Return an empty string or apply other suitable logic
            
            sentence=str(sentence) #convert to strings series
            sentence = sentence.lower()  # lowercase everything 
            sentence = re.sub(r'^rt[\s]+', '', sentence) # Remove "RT" (retweet abbreviation)
            sentence = re.sub(r'@\w+', '', sentence) # Remove mentions (@username)
            sentence = re.sub(r'https?://\S+', '', sentence) # Remove URLs
            sentence = re.sub(r'\[.*?\]', '', sentence) # remove square brackets
            sentence = re.sub(r'<.*?>+', '', sentence) # remove angular brackets
            sentence = re.sub(r'\d+', '', sentence) #remove numbers
            sentence = re.sub(r'\w*\d\w*', '', sentence) #remove words containing numbers
            sentence = re.sub(r'\b(\w)\b', '', sentence)  # Remove single characters
            
            # Remove special characters and hashtags
            sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence)
            sentence = re.sub(r'#', '', sentence)
            
            sentence = sentence.strip() # remove leading and trailing white spaces

            # Tokenize sentence for BERT
            tokens = self.tokenizer.tokenize(sentence)
            sentence = ' '.join(tokens)
            
            log.info("Main cleaning successful.")
            return sentence
            
        except Exception as e:
            log.error("Error in cleaning the sentence.")
            raise CustomException(e, sys)
        
        
    def clean_file(self) -> None:
        """Cleans an input file and writes the cleaned text to an output file."""
        log.info("Starting file cleaning process")
        try:
            input_filepath = os.path.join(self.config.source_dir, self.config.input_file)
            output_filepath = os.path.join(self.config.destination_dir, self.config.output_file)
            
            df = pd.read_csv(input_filepath)
            log.info(f"Loaded data from {input_filepath}")
            log.info(f"Initial DataFrame shape: {df.shape}")
            
            # Remove columns 'image_damage' and 'image_damage_conf' if they exist
            columns_to_drop = ['image_damage', 'image_damage_conf']
            for column in columns_to_drop:
                if column in df.columns:
                    df.drop(column, axis=1, inplace=True)
                    log.info(f"Removed column: {column}")
                else:
                    log.warning(f"Column '{column}' not found in dataset.")
            
            # Check for missing values in the expected text column 
            if df['tweet_text'].isnull().any():
                missing_count = df['tweet_text'].isnull().sum()
                log.error(f"Found {missing_count} missing values in the 'tweet_text' column. Aborting the cleaning process.")
                raise CustomException(f"Missing values detected in 'text' column: {missing_count}", sys)
            
            
            if 'tweet_text' not in df.columns:
                raise CustomException("Input file does not contain a 'tweet_text' column", sys)
            
            
            # Apply the cleaning function to the tweet text column
            df['cleanText'] = df['tweet_text'].apply(self.clean)
            
            log.info(f"Cleaned DataFrame shape: {df.shape}")
            
            # Write cleaned data to the output file
            df.to_csv(output_filepath, index=False)
            log.info(f"Cleaned data written to {output_filepath}")
            
            log.info("File cleaning process successful.")
                    
        except Exception as e:
            log.error(f"Error occurred while cleaning data from {input_filepath} to {output_filepath}")
            raise CustomException(e, sys)