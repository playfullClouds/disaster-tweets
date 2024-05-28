import re
import gc
import os
import sys
import nltk
import emoji
import pandas as pd


from autocorrect import Speller
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('all', quiet=True)


from src.logger import log
from dataclasses import dataclass, field
from src.exception import CustomException



@dataclass
class DataCleaningConfig:
    source_dir: str = os.path.join('artifacts', 'data_ingestion')
    destination_dir: str = os.path.join('artifacts', 'data_cleaner')
    input_file: str = 'tweets.csv'
    output_file: str = field(default='cleaned_data.csv')
    

class DataCleaner:
    def __init__(self) -> None:
        log.info("Initializing DataCleaner")
        self.config = DataCleaningConfig()

        try:
            # Create only the base directory since the zip file path is part of this directory.
            os.makedirs(self.config.destination_dir, exist_ok=True)
            log.info(f"Created directory {self.config.destination_dir}")

            log.info("Data Cleaning directory setup completed")
        except Exception as e:
            log.error("Failed to set up the data cleaning directories.")
            raise CustomException
        
        # Initialize NLP tools and stopwords
        log.info("Initializing NLP tools and stopwords")
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stemmer = PorterStemmer()
            self.tt = TweetTokenizer()
            self.spell = Speller(lang='en')
            
            # Extend stopwords with custom words
            self.stop_words = stopwords.words('english')
            self.stop_words.extend([
                'rt', 'via', '...', 'u', 'im', 'ur', 'rt', 'https', 'co', 'amp', 'rt', 
                '@', 'url', '—', '•', '➡', 'http?', 'https?', '…', '️', '🙏', '❤️'
            ])
            log.info("NLP tools and stopwords initialized successfully")
        except Exception as e:
            log.error("Failed to initialize NLP tools and stopwords")
            raise CustomException(e, sys)
        
    
    def extract_and_remove_emojis(self, text: str) -> str:
        """Removes emojis from the given text and returns a text without emojis."""
        log.info("Extracting and removing emoji starting")
        try:
            # Extract emoji list from the text
            emoji_list = emoji.emoji_list(text)
            extracted_emojis = [item['emoji'] for item in emoji_list]
            log.info(f"Extracted emojis: {extracted_emojis}")

            # Create a regex pattern to match the emojis
            emoji_pattern = re.compile('|'.join(map(re.escape, extracted_emojis)))
            
            log.info("Extracting and removing emoji successfully")
            return emoji_pattern.sub(r'', text)
        
        except Exception as e:
            log.error("Error in extracting and removing emojis")
            raise CustomException(e, sys)
            
    
    def clean(self, sentence: str) -> str:
        """Cleans a single sentence by removing unwanted tokens and applying normalization."""
        log.info("Main cleaning starting")
        try:
            
            # If the sentence is None or not a string, handle it gracefully
            if not sentence or not isinstance(sentence, str):
                log.warning(f"Invalid sentence value encountered: {sentence}")
                return ""  # Return an empty string or apply other suitable logic
            
            sentence=str(sentence) #convert to strings series
            sentence = sentence.lower()  # lowercase everything 
            sentence = self.extract_and_remove_emojis(sentence)  # Remove emojis
            sentence = sentence.encode('ascii', 'ignore').decode() #encode ASCII characters
            sentence = re.sub(r'^rt[\s]+', '', sentence) # Remove "RT" (retweet abbreviation)
            sentence = re.sub(r'@\w+', '', sentence) # Remove mentions (@username)
            sentence = re.sub(r'https?://\S+', '', sentence) # Remove URLs
            sentence = re.sub(r'\[.*?\]', '', sentence) # remove square brackets
            sentence = re.sub(r'<.*?>+', '', sentence) # remove angular brackets
            sentence = re.sub(r'\d+', '', sentence) #remove numbers
            sentence = re.sub(r'\w*\d\w*', '', sentence) #remove words containing numbers
            sentence = sentence.strip() # remove leading and trailing white spaces


            # correct sentence and filter out word with two letter or less
            sentence_corrected = ' '.join([self.spell(word) for word in sentence.split() if len(word) > 2]) 


            tokens = self.tt.tokenize(sentence_corrected) # tokenize with TweetTokenizer
            filtered_words = [w for w in tokens if not w in self.stop_words] # remove stopwords 
            
            # Apply stemming and lemmatization
            stem_words=[self.stemmer.stem(w) for w in filtered_words] #Stemming the words         
            lemma_words=[self.lemmatizer.lemmatize(w) for w in stem_words] #Lemmatization
            
 
            log.info("Main cleaning successful.")
            return " ".join(lemma_words)
        
        except Exception as e:
            log.error("Error in cleaning the sentence.")
            raise CustomException(e, sys)
        
    
    def clean_file(self) -> None:
        """Cleans an input file and writes the cleaned text to an output file."""
        log.info("Main cleaning starting")
        try:
            
            output_filepath = os.path.join(self.config.destination_dir, self.config.output_file)
            
            
            # Check if the cleaned data file already exists
            if os.path.exists(output_filepath):
                log.info(f"Cleaned data file {output_filepath} already exists. Skipping data cleaning.")
                return
            
            input_filepath = os.path.join(self.config.source_dir, self.config.input_file)
            
            
            df = pd.read_csv(input_filepath)
            log.info(f"Loaded data from {input_filepath}")
            
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
            
            # Write cleaned data to the output file
            df.to_csv(output_filepath, index=False)
            log.info(f"Cleaned data written to {output_filepath}")
            
            
            # Release memory
            del df
            gc.collect()
            
            log.info("Main cleaning successful.")
                    
        except Exception as e:
            log.error(f"Error occurred while cleaning data from {input_filepath} to {output_filepath}")
            raise CustomException(e, sys)
    
    
    
    
# if __name__ == "__main__":
#     cleaner = DataCleaner()
#     cleaner.clean_file()
    

STAGE_NAME = "Data Cleaning"
try:
   log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   cleaner = DataCleaner()
   cleaner.clean_file()
   log.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        log.exception(f"Exception occurred during {STAGE_NAME}")
        raise CustomException(e, sys)    







# stk = stopwords.words('english')
# len(stk)


# # add words to stopwords

# stk.extend([
#     'rt', 'via', '...', 'u', 'im', 'ur', 'u\'re', 'c', 'b', 'don',
#     'amp', 'RT', '@', 'rt', 'http', 'https', 'co', 'now', 'should', 'just',
#     'rts', 'retweet', 'retweets', '…', '️', 'cc' ,'a', 'an', 'the', 'and', 'or', 
#     'but', 'if', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
#     'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 
#     'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 
#     'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
#     'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 
#     'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
#     't', 'can', 'will', 'rt', 'via', '’', '‘', '“', '”', '–', '—', '•', '➡', 'http?', 'https?',
#     '➡️', '⬅', '⬅️', '👉', '👇', '👍', '🙏', '❤️', '🔥',  'u', '2', '4', 'im', 'ur',
#     'AT_USER', 'URL',  '’s', '...!', '...?', '&amp;', 'amp;', 'amp', '✈️', '✡', '✨', '❤', 
#     '||', '|'


# ])


