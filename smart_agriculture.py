"""Smart agriculture crop recommendation with live weather integration."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from weather_api import fetch_live_weather

warnings.filterwarnings("ignore")

DATA_PATH = Path("data/Crop_recommendation.csv")
MODEL_PATH = Path("crop_model.joblib")
FEATURES = ["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall"]
TARGET = "Crop"
RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = FEATURES + [TARGET]
    df[TARGET] = df[TARGET].str.strip().str.title()
    return df


def train_model(df: pd.DataFrame) -> tuple[Pipeline, LabelEncoder, float, float]:
    """Train a tuned Random Forest model and return metrics."""
    X = df[FEATURES]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_split=2,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    cv_accuracy = cross_val_score(model, X_train, y_train, cv=5).mean()
    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    artifact = {
        "model": model,
        "label_encoder": label_encoder,
        "features": FEATURES,
        "defaults": df[FEATURES].median().to_dict(),
    }
    joblib.dump(artifact, MODEL_PATH)

    return model, label_encoder, cv_accuracy, test_accuracy


def load_or_train_model() -> tuple[Pipeline, LabelEncoder, dict, float, float]:
    df = load_dataset()
    if MODEL_PATH.exists():
        artifact = joblib.load(MODEL_PATH)
        model = artifact["model"]
        label_encoder = artifact["label_encoder"]
        defaults = artifact["defaults"]
        X = df[FEATURES]
        y = label_encoder.transform(df[TARGET])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        cv_accuracy = cross_val_score(model, X_train, y_train, cv=5).mean()
        test_accuracy = accuracy_score(y_test, model.predict(X_test))
        return model, label_encoder, defaults, cv_accuracy, test_accuracy

    model, label_encoder, cv_accuracy, test_accuracy = train_model(df)
    defaults = df[FEATURES].median().to_dict()
    return model, label_encoder, defaults, cv_accuracy, test_accuracy


def analyze_conditions(temperature: float, humidity: float, soil_moisture_pct: float | None, rainfall: float):
    if soil_moisture_pct is None:
        if humidity > 75 and rainfall > 50:
            soil_status, irrigation = "Wet Soil", "No Irrigation Required"
        elif humidity < 45 or rainfall < 20:
            soil_status, irrigation = "Dry Soil", "Irrigation Required"
        else:
            soil_status, irrigation = "Moderate Moisture", "Normal Irrigation"
    elif soil_moisture_pct < 25:
        soil_status, irrigation = "Dry Soil", "Irrigation Required"
    elif soil_moisture_pct < 45:
        soil_status, irrigation = "Moderate Moisture", "Normal Irrigation"
    else:
        soil_status, irrigation = "Wet Soil", "No Irrigation Required"

    if humidity > 80 and rainfall > 100:
        rain_alert = "High Chance of Rainfall"
    elif rainfall > 50:
        rain_alert = "Moderate Rainfall Expected"
    else:
        rain_alert = "Normal Weather Conditions"

    if temperature > 35:
        temp_status = "High Temperature"
    elif temperature < 15:
        temp_status = "Low Temperature"
    else:
        temp_status = "Moderate Temperature"

    return soil_status, irrigation, rain_alert, temp_status


def predict_from_weather(
    city: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    nitrogen: float | None = None,
    phosphorus: float | None = None,
    potassium: float | None = None,
    ph: float | None = None,
) -> dict:
    """Fetch live weather and recommend the best crop."""
    weather = fetch_live_weather(city=city, latitude=latitude, longitude=longitude)
    model, label_encoder, defaults, cv_accuracy, test_accuracy = load_or_train_model()

    sample = pd.DataFrame(
        [
            {
                "N": nitrogen if nitrogen is not None else defaults["N"],
                "P": phosphorus if phosphorus is not None else defaults["P"],
                "K": potassium if potassium is not None else defaults["K"],
                "Temperature": weather["temperature"],
                "Humidity": weather["humidity"],
                "pH": ph if ph is not None else defaults["pH"],
                "Rainfall": weather["rainfall"],
            }
        ]
    )

    prediction_idx = model.predict(sample[FEATURES])[0]
    probabilities = model.predict_proba(sample[FEATURES])[0]
    crop = label_encoder.inverse_transform([prediction_idx])[0]
    confidence = probabilities[prediction_idx] * 100

    soil_status, irrigation, rain_alert, temp_status = analyze_conditions(
        weather["temperature"],
        weather["humidity"],
        weather["soil_moisture_pct"],
        weather["rainfall"],
    )

    prob_df = pd.DataFrame(
        {
            "Crop": label_encoder.classes_,
            "Probability (%)": probabilities * 100,
        }
    ).sort_values("Probability (%)", ascending=False)

    return {
        "weather": weather,
        "inputs": sample.iloc[0].to_dict(),
        "recommended_crop": crop,
        "confidence": confidence,
        "probabilities": prob_df,
        "soil_status": soil_status,
        "irrigation": irrigation,
        "rain_alert": rain_alert,
        "temperature_status": temp_status,
        "cv_accuracy": cv_accuracy * 100,
        "test_accuracy": test_accuracy * 100,
    }


def print_prediction(result: dict) -> None:
    weather = result["weather"]
    print("=" * 55)
    print("     SMART AGRICULTURE - LIVE WEATHER PREDICTION")
    print("=" * 55)
    print(f"Location         : {weather['location']}")
    print(f"Weather Time     : {weather['time']}")
    print(f"Temperature      : {weather['temperature']} °C")
    print(f"Humidity         : {weather['humidity']} %")
    print(f"Rainfall (24h)   : {weather['rainfall']} mm")
    if weather["soil_moisture_pct"] is not None:
        print(f"Soil Moisture    : {weather['soil_moisture_pct']} %")
    print("-" * 55)
    print(f"Model CV Accuracy : {result['cv_accuracy']:.2f}%")
    print(f"Model Test Accuracy: {result['test_accuracy']:.2f}%")
    print("-" * 55)
    print(f"Recommended Crop : {result['recommended_crop']}")
    print(f"Confidence       : {result['confidence']:.2f}%")
    print(f"Soil Condition   : {result['soil_status']}")
    print(f"Irrigation       : {result['irrigation']}")
    print(f"Rain Alert       : {result['rain_alert']}")
    print(f"Temperature      : {result['temperature_status']}")
    print("=" * 55)
    print("\nTop Crop Probabilities:")
    print(result["probabilities"].head().to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Live weather crop recommendation")
    parser.add_argument("--city", default="Hyderabad", help="City name for live weather")
    parser.add_argument("--latitude", type=float, help="Latitude override")
    parser.add_argument("--longitude", type=float, help="Longitude override")
    parser.add_argument("--nitrogen", type=float, help="Soil nitrogen (N)")
    parser.add_argument("--phosphorus", type=float, help="Soil phosphorus (P)")
    parser.add_argument("--potassium", type=float, help="Soil potassium (K)")
    parser.add_argument("--ph", type=float, help="Soil pH value")
    args = parser.parse_args()

    result = predict_from_weather(
        city=args.city if args.latitude is None else None,
        latitude=args.latitude,
        longitude=args.longitude,
        nitrogen=args.nitrogen,
        phosphorus=args.phosphorus,
        potassium=args.potassium,
        ph=args.ph,
    )
    print_prediction(result)


if __name__ == "__main__":
    main()
