import pandas as pd
import tensorflow as tf
from flask import Flask, Response
import prometheus_client
from prometheus_client import Gauge

# Load trained ML model
model = tf.keras.models.load_model("project.keras")

# Load dataset from CSV
df = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
print(df.columns)  # Debugging: Print column names
 
expected_columns = ['Flow Duration', 'Total Fwd Packets', 'Total Bwd Packets', 'Flow Bytes/s']
available_columns = [col for col in expected_columns if col in df.columns]

x = df[available_columns]  # Select only available columns
x = df.iloc[:, :78]  # Select the first 78 columns
print(x.shape)  
# Create a Prometheus metric for predictions
ddos_prediction = Gauge('ddos_predictions', 'DDoS attack probability')

app = Flask(__name__)
@app.route('/')
def home():
    return "Flask is running! Use /metrics for Prometheus data."
@app.route('/metrics')
def metrics():
    # Convert first row of dataset into model input
    print(x.shape)
    input_data = x.iloc[:10, :].values.reshape(1, 10, 78)  
      # Adjust for time steps & features  

    print("Input shape:", input_data.shape)
    print(model.input_shape)

    prediction_value = model.predict(input_data)[0][0]  # Predict attack probability
    ddos_prediction.set(prediction_value)  # Send prediction to Prometheus

    return Response(prometheus_client.generate_latest(), mimetype="text/plain")


if __name__ == '__main__':
    app.run(debug = True, host='0.0.0.0', port=5000)
    model.summary()