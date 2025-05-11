Neural IDS – Real-time DDoS Detection Using Deep Learning

This project implements a deep learning-based Intrusion Detection System (IDS) to detect DDoS attacks in real time. It uses a hybrid CNN-LSTM model, a Flask API for predictions, Prometheus for scraping metrics, and Grafana for real-time visualization.

Technologies Used
- **Python 3.11**
- **TensorFlow / Keras**
- **Flask**
- **Prometheus**
- **Grafana**
- **Pandas**, **NumPy**, **Matplotlib**

 Project Structure

project/
│
├── app.py                # Flask API exposing DDoS prediction
├── project.ipynb         # Jupyter Notebook (Model training & analysis)
├── project.keras         # Saved CNN-LSTM model
│
├── prometheus/           # Prometheus setup
│   ├── prometheus.exe
│   └── prometheus.yml
│
└── grafana/              # Grafana setup
    └── bin/
        └── grafana-server.exe

 How to Run the Project

 1. Install Required Python Libraries
```bash
pip install flask tensorflow pandas prometheus_client
```
> **Note:** Ensure Python version is **3.10** or **3.11** for TensorFlow compatibility.

 2. Run the Flask Prediction Server
```bash
cd C:\Users\<YourUsername>\Desktop\project
python app.py
```
- Access metrics at: [http://localhost:5000/metrics](http://localhost:5000/metrics)


 3. Start Prometheus
```bash
cd C:\Users\<YourUsername>\Desktop\project\prometheus
prometheus.exe --config.file=prometheus.yml
```
- Visit Prometheus targets: [http://localhost:9090/targets](http://localhost:9090/targets)

 4. Start Grafana
```bash
cd C:\Users\<YourUsername>\Desktop\project\grafana\bin
grafana-server
```
- Open Grafana in browser: [http://localhost:3000](http://localhost:3000)
- Default login: `admin / admin` (you will be prompted to change this on first login)


## Grafana Dashboard Setup
1. Add **Prometheus** as a data source: `http://localhost:9090`
2. Create a panel using the metric: `ddos_predictions`
3. (Optional) Set alerts if predictions exceed a threshold (e.g., `0.8`)


## Model Summary
- **Architecture**: CNN + LSTM
- **Input**: Processes 10 time steps with 78 features per input
- **Dataset**: Trained on the **CICIDS dataset** (DDoS subset)



## Disclaimer
This project is for educational and research purposes only. Use responsibly in secure and ethical environments.
