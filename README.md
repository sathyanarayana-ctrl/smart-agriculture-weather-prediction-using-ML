# Smart Agriculture Weather Prediction using ML

A machine learning project that recommends the best crop based on weather and soil conditions.

## Features

- **Multi-model comparison**: Decision Tree, Random Forest, KNN, and SVM
- **Cross-validation** for reliable model selection
- **Feature scaling** via StandardScaler pipeline
- **Crop prediction** with confidence scores
- **Soil, rainfall, and temperature analysis**
- **EDA visualizations**: box plots, correlation heatmap, dashboard

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn seaborn jupyter
```

## Usage

Open and run `project_code.ipynb` in Jupyter Notebook or Google Colab.

For interactive input, uncomment the input cell at the bottom of the notebook.

## Input Parameters

| Parameter     | Description              |
|---------------|--------------------------|
| Temperature   | Air temperature (°C)     |
| Humidity      | Relative humidity (%)    |
| Soil_Moisture | Soil moisture level (%)  |
| Water_Level   | Water table level        |

## Output

- Recommended crop with confidence percentage
- Soil condition and irrigation advice
- Rainfall alert
- Temperature status
- Classification metrics and visualizations
