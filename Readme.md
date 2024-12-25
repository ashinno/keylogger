# Keystroke Anomaly Detection

## Overview
This project captures keystroke data, extracts timing features, and uses machine learning (Isolation Forest) to detect anomalies in typing behavior. 

## Features
- Captures and records keystrokes (press/release timing).
- Encrypts and securely stores the keystroke data.
- Extracts features such as hold time and flight time.
- Trains an Isolation Forest model for anomaly detection.
- Real-time anomaly detection with alerts.

## Requirements
- Python 3.x
- pynput
- cryptography
- scikit-learn
- numpy
- joblib

## Installation
```bash
pip install pynput cryptography scikit-learn numpy joblib
```

## Usage
Run the Jupyter Notebook to start capturing keystrokes:
```bash
jupyter main.ipynb
```
Press `ESC` to stop the listener. Detected anomalies will be printed in the console.

## Notes
- This is for educational purposes. Handle keystroke data responsibly and ensure compliance with privacy regulations.
- Modify the 'esc' key stop condition for production use.

## License
MIT License

