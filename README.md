# Smart Agriculture Weather Prediction using ML

A machine learning project that connects to **live weather data** and recommends the best crop based on real-time conditions and soil parameters.

## Features

- **Live weather API** — Open-Meteo (no API key required)
- **High-accuracy model** — ~99.5% test accuracy on 2,200 crop samples
- **22 crop types** — Rice, Wheat, Maize, Cotton, and more
- **Confidence scores** and weather alerts
- **Custom soil input** — optional N, P, K, pH values

## Requirements

```bash
pip install -r requirements.txt
```

## Quick Start (Command Line)

```bash
# Predict crop for Hyderabad using live weather
python smart_agriculture.py --city Hyderabad

# With custom soil values
python smart_agriculture.py --city Mumbai --nitrogen 90 --phosphorus 42 --potassium 43 --ph 6.5
```

## Jupyter Notebook

Open and run `project_code.ipynb` in Jupyter or Google Colab.

## How It Works

1. Fetches live **temperature**, **humidity**, **rainfall**, and **soil moisture** from Open-Meteo
2. Combines weather with soil data (defaults or user-provided N, P, K, pH)
3. Trains a Random Forest model on 2,200 real crop recommendation samples
4. Returns the best crop with confidence % and weather alerts

## Model Accuracy

| Metric | Value |
|--------|-------|
| Cross-validation | ~99.49% |
| Test set | ~99.55% |

> **Note:** 99.99% accuracy is not realistic for real-world weather prediction. The model achieves ~99.5% on the standard crop recommendation dataset. Live prediction confidence depends on how closely your soil matches the training data.

## Input Parameters

| Parameter | Source |
|-----------|--------|
| Temperature | Live weather API |
| Humidity | Live weather API |
| Rainfall | Live weather API (24h forecast) |
| Soil Moisture | Live weather API |
| N, P, K, pH | User input or dataset defaults |

## Output

- Recommended crop with confidence percentage
- Soil condition and irrigation advice
- Rainfall and temperature alerts
- Top crop probability rankings
