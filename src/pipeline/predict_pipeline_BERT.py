import os
# Set the environment variable to disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import os
import sys
import pandas as pd
import tensorflow as tf


from transformers import BertTokenizer, TFBertForSequenceClassification

from src.logger import log
from src.exception import CustomException


class PredictPipeline:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Paths to the saved model dir
        self.model_dir = os.path.join(BASE_DIR, 'artifacts', 'model_trainer', 'bert_model')
        
        # # Set HF_HOME environment variable
        # os.environ["HF_HOME"] = os.path.join(BASE_DIR, "artifacts", "model_trainer", "cache")

        log.info("Loading BERT tokenizer and model from saved directory.")
        log.info(f"Model directory path: {self.model_dir}")
        
        try:
            # List contents of the model directory for debugging
            log.info(f"Contents of the model directory: {os.listdir(self.model_dir)}")
            
            # Load the tokenizer from the saved directory
            self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
            # Load the model from the saved directory
            self.model = TFBertForSequenceClassification.from_pretrained(self.model_dir)
            log.info("Model and tokenizer successfully loaded.")
        except Exception as e:
            log.error("Error loading model and tokenizer from saved directory.")
            raise CustomException(e, sys)
    
        
        # Initialize the BERT tokenizer
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        
    def predict(self, text_data):
        log.info("Loading model and vectorizer.")
        
        try:
            
            # Ensure text_data is in the expected format (e.g., a string)
            if isinstance(text_data, str):
                text_data = [text_data]
                
            log.info("Tokenizing text data.")
            
             # Tokenize the input text data
            inputs = self.tokenizer(text_data, padding=True, truncation=True, return_tensors="tf")
            
            log.info("Making predictions.")
            
            # Predict using the loaded model
            outputs = self.model(inputs)
            logits = outputs.logits
            preds = tf.argmax(logits, axis=1).numpy().tolist()
            
            log.info("Predictions successfully made.")
            return preds
            
        except Exception as e:
            log.error("Error during prediction.")
            raise CustomException(e, sys)
    

class CustomData:
    def __init__(self, tweet_text: str):
        self.tweet_text = tweet_text
        
    def get_data_as_data_frame(self):
        log.info("Creating DataFrame from custom input data.")
        
        try:
            # Create a DataFrame containing the tweet text
            data_dict = {"tweet_text": [self.tweet_text]}
            return pd.DataFrame(data_dict)
        
        except Exception as e:
            log.error("Error in creating DataFrame from custom input data.")
            raise CustomException(e, sys)