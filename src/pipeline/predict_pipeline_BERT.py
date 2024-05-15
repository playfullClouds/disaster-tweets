import os
import sys
import torch
import pandas as pd


from transformers import BertTokenizer, BertForSequenceClassification

from src.logger import log
from src.exception import CustomException


class PredictPipeline:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Paths to the saved model and vectorizer
        # self.model_path = os.path.join(BASE_DIR, 'artifacts', 'model_trainer', 'best_model.pkl')
        # Path to the saved model
        self.model_dir = os.path.join(BASE_DIR, 'artifacts', 'model_trainer')
        self.model_path = os.path.join(self.model_dir, 'pytorch_model.bin')
        
        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def predict(self, text_data):
        log.info("Loading model and vectorizer.")
        
        try:
            # Load the trained model
            model = BertForSequenceClassification.from_pretrained(self.model_dir)
            model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
            model.eval()
            
            # Ensure text_data is in the expected format (e.g., a string)
            if isinstance(text_data, str):
                text_data = [text_data]
                
            log.info("Tokenizing text data.")
            
            # Tokenize the input text data
            inputs = self.tokenizer(text_data, padding=True, truncation=True, return_tensors="pt")
            
            log.info("Making predictions.")
            
            # Predict using the loaded model
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).tolist()
                
            log.info("Predictions successfully made.")
            return preds
            
        except Exception as e:
            log.error("Error in loading model and vectorizer")
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