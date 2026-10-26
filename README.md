# 🚗 Used Car Price Prediction System

A machine learning web application that predicts the estimated selling price of a used car based on its characteristics. Built using Python, Scikit-learn, FastAPI, HTML, CSS, and JavaScript, the application combines a trained regression model with a responsive web interface.

## Features

- **Price Prediction:** Estimates used-car selling prices using a trained machine learning model.
- **Interactive Web Interface:** Enter car details through a clean, responsive interface.
- **Categorical Feature Encoding:** Uses One-Hot Encoding to process categorical features such as car name, fuel type, and transmission.
- **Machine Learning Pipeline:** Integrates preprocessing and prediction into a reusable Scikit-learn pipeline.
- **REST API:** Exposes prediction functionality through FastAPI endpoints.
- **Dynamic Dropdowns:** Loads available car models and other categorical options from the backend.
- **API Documentation:** Provides interactive API documentation through Swagger UI.

## Technology Stack

- **Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Backend:** FastAPI, Uvicorn
- **Frontend:** HTML, CSS, JavaScript
- **Model Serialization:** Joblib

## Machine Learning Workflow

1. Load and explore the used-car dataset.
2. Clean duplicate records and handle missing values.
3. Separate input features from the target selling price.
4. Split the data into training and testing sets.
5. Apply numerical scaling and categorical encoding.
6. Train a Linear Regression model.
7. Evaluate predictions using MAE, RMSE, and R².
8. Save the trained pipeline and integrate it with FastAPI.
9. Display predictions through the web interface.

## Input Features

The model uses the following features:

- Car name
- Manufacturing year
- Kilometres driven
- Fuel type
- Seller type
- Transmission type
- Ownership category

**Target variable:** Selling price.

## Installation and Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd used-car-price-predictor
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install fastapi "uvicorn[standard]" scikit-learn joblib pandas numpy
```

### 4. Add the trained model files

Ensure these files are present in the project directory:

- `car_price_model.joblib`
- `car_metadata.json`

The model file should be generated from the training notebook, and the metadata file should contain the corresponding categorical options and supported input ranges.

### 5. Run the application

```bash
uvicorn main:app --reload
```

### 6. Open the application

- **Web Interface:** http://127.0.0.1:8000
- **API Documentation:** http://127.0.0.1:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the web interface |
| GET | `/options` | Returns available input options |
| POST | `/predict` | Predicts the estimated selling price |
| GET | `/docs` | Opens interactive API documentation |

## Model Evaluation

The model is evaluated using:

- **Mean Absolute Error (MAE):** Measures average absolute prediction error.
- **Root Mean Squared Error (RMSE):** Penalizes large prediction errors more heavily.
- **R² Score:** Measures performance relative to predicting the mean target value.

Actual evaluation results should be reported after running the model on the held-out test dataset.

## Future Improvements

- Compare Linear Regression with Random Forest and Decision Tree regressors.
- Perform cross-validation and hyperparameter tuning.
- Incorporate additional features such as vehicle condition and location.
- Retrain using more recent and representative market data.
- Deploy the application to a cloud hosting platform.
- Add automated tests, logging, and prediction monitoring.
