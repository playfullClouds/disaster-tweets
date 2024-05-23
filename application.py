import os
import numpy as np

from flask import Flask, request, render_template
from src.pipeline.predict_pipeline_BERT import CustomData, PredictPipeline


# Set the environment variable to disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Intialize Flask app
application = Flask(__name__)
app = application

@app.route('/')
def index():
    # Render the landing page
    return render_template('index.html')


@app.route('/predict', methods=['GET'])
def predict():
    # Render the prediction input page
    return render_template('home.html')

@app.route('/predictdata', methods=['POST'])
def predict_data():
    # Extract from data, perform prediction, and show results
    tweet_text = request.form.get('tweet_text')
    
    # Use the CustomData class to wrap the input data
    data = CustomData(tweet_text=tweet_text)
    
    # Convert to a DataFrame or suitable format required for prediction
    pred_df = data.get_data_as_data_frame()
    
    # Initialize the prediction pipeline and get results
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df['tweet_text'].tolist())
    
    # Extract the first prediction (if there's more that one) and determine the classification result
    if isinstance(results, np.ndarray):
        result = results[0]
    else:
        result = results
        
    # Format the result for the user-friendly display
    formatted_result = "Informative" if result == 1 else "Not Informative"
    
    # Pass the formatted result to my template
    return render_template('result.html', results=formatted_result)


if __name__ == "__main__":
    app.run(debug=True)

