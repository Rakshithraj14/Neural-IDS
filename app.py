import pandas as pd
import tensorflow as tf
from flask import Flask, Response
import prometheus_client
from prometheus_client import Gauge
import numpy as np
import logging
import sys

# Configure basic logging
logging.basicConfig(stream=sys.stdout, 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load trained ML model
model = None # Initialize model to None
try:
    model = tf.keras.models.load_model("project.keras")
    logging.info("Model loaded successfully.")
except FileNotFoundError:
    logging.error("Model file 'project.keras' not found.")
except IOError as e: # Catches other I/O errors like permission issues
    logging.error(f"IOError loading model 'project.keras': {e}")
except Exception as e:
    logging.error(f"Unexpected error loading model: {e}")
    # model remains None

# Load and preprocess dataset at startup
X_processed = np.array([]) # Initialize as empty array
try:
    df = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    # Preprocessing steps from the notebook
    df.dropna(axis=1, how='all', inplace=True) 
    df.drop(columns=[col for col in df.columns if 'Unnamed' in col], inplace=True) 
    df.replace([np.inf, -np.inf], np.nan, inplace=True) 
    df.dropna(inplace=True) 
    df.rename(columns=lambda x: x.strip(), inplace=True)

    # Prepare data for the model
    X = df.drop('Label', axis=1, errors='ignore') 
    X = X.select_dtypes(include=[np.number]) 
    X_processed = X.iloc[:, :78].values 
    if X_processed.size == 0:
        logging.warning("Dataset loaded but resulted in empty processed features after preprocessing.")
    else:
        logging.info(f"Dataset loaded and preprocessed successfully. Shape of processed features: {X_processed.shape}")
except FileNotFoundError:
    logging.error("Dataset file 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv' not found.")
except pd.errors.EmptyDataError:
    logging.error("Dataset file is empty or unreadable.")
except Exception as e:
    logging.error(f"Error loading or preprocessing dataset: {e}")
    # X_processed remains np.array([])

# Create a Prometheus metric for predictions
ddos_prediction = Gauge('ddos_predictions', 'DDoS attack probability')

app = Flask(__name__)

@app.route('/')
def home():
    return "Flask is running! Use /metrics for Prometheus data."

@app.route('/metrics')
def metrics():
    logging.info("/metrics endpoint called")
    global X_processed 

    if model is None:
        logging.error("Model not loaded. Cannot make predictions.")
        return Response("Internal server error: Model not available", status=500, mimetype="text/plain")

    if X_processed.size == 0:
        logging.error("Processed data is not available. Cannot make predictions.")
        return Response("Internal server error: Data not available", status=500, mimetype="text/plain")

    # Select 10 random rows from the preprocessed data
    if X_processed.shape[0] >= 10:
        random_indices = np.random.choice(X_processed.shape[0], size=10, replace=False)
        input_data_sample = X_processed[random_indices, :]
    else:
        # Fallback if less than 10 rows are available: use all available and pad with zeros
        logging.warning(f"Not enough data for 10 random samples (available: {X_processed.shape[0]}). Using available data with padding if necessary.")
        input_data_sample = X_processed
        if X_processed.shape[0] < 10: 
            padding_shape = (10 - X_processed.shape[0], X_processed.shape[1] if X_processed.ndim > 1 and X_processed.shape[1] > 0 else 78)
            padding = np.zeros(padding_shape)
            if input_data_sample.ndim == 1 and input_data_sample.size > 0 : 
                input_data_sample = input_data_sample.reshape(1, -1)
            
            if X_processed.shape[0] == 0: # Should be caught by X_processed.size == 0 earlier, but defensive
                 input_data_sample = padding
            elif X_processed.shape[0] < 10 : 
                 input_data_sample = np.vstack((input_data_sample, padding))
    
    input_data_reshaped = input_data_sample.reshape(1, 10, 78)
    logging.info(f"Predicting with input shape: {input_data_reshaped.shape}")

    try:
        prediction_value = model.predict(input_data_reshaped)[0][0]
        logging.info(f"Raw prediction value: {prediction_value}")
        ddos_prediction.set(prediction_value)
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return Response(f"Internal server error: Prediction failed ({e})", status=500, mimetype="text/plain")

    return Response(prometheus_client.generate_latest(), mimetype="text/plain")

if __name__ == '__main__':
    logging.info("Flask application starting...")
    app.run(debug=True, host='0.0.0.0', port=5000)
