import os
import sys
import pandas as pd


from src.logger import log
from src.utils import load_object
from src.exception import CustomException


class PredictPipeline:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Paths to the saved model and vectorizer
        # self.model_path = os.path.join(BASE_DIR, 'artifacts', 'model_trainer', 'best_model.pkl')
        # self.model_path = os.path.join(BASE_DIR, 'artifacts', 'model_trainer', 'best_model_SMOTE.pkl')
        self.model_path = os.path.join(BASE_DIR, 'artifacts', 'model_optimizer', 'optimized_model_SMOTE.pkl')
        
        # self.model_path = os.path.join(BASE_DIR, 'artifacts', 'model_trainer', 'best_model_SMOTE.pkl')
        self.preprocessor_path = os.path.join(BASE_DIR, 'artifacts', 'data_transformation', 'tfidf_vectorizer_SMOTE.pkl')
        self.scaler_path = os.path.join(BASE_DIR, 'artifacts', 'data_transformation', 'scaler_SMOTE.pkl')
        
    def predict(self, text_data):
        log.info("Loading model, vectorizer, and scaler.")
        
        try:
            # Load the trained model and TfidfVectorizer object
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)
            scaler = load_object(file_path=self.scaler_path)
            
            # Ensure text_data is in the expected format (e.g., a string)
            if isinstance(text_data, str):
                text_data = [text_data]
                
            log.info("Transforming text data.")
            
            # Transform the input text data
            data_transformed = preprocessor.transform(text_data)
            
            # Scale the transformed text data
            data_scaled = scaler.transform(data_transformed.toarray())
            
            log.info("Making predictions.")
            
            # Predict using the trained model
            preds = model.predict(data_scaled)
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