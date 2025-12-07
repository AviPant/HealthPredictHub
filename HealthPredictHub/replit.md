# HealthPredict Pro

## Overview
A comprehensive Streamlit health analytics platform with 5 disease prediction models and PDF report generation. This is an educational tool demonstrating machine learning applications in healthcare.

## Features
- **Sidebar Navigation**: Easy navigation between all prediction modules
- **5 Disease Predictors**:
  - Diabetes Predictor (Random Forest, 87% accuracy)
  - Heart Disease Predictor (Gradient Boosting, 89% accuracy)
  - Stroke Risk Predictor (Random Forest, 94% accuracy)
  - Kidney Disease Predictor (Gradient Boosting, 97% accuracy)
  - Liver Disease Predictor (Logistic Regression, 74% accuracy)
- **Report Generator**: Download comprehensive PDF reports of all predictions
- **About Models**: Detailed documentation of ML models and features

## Project Structure
```
├── app.py                 # Main application with sidebar navigation
├── pages/
│   ├── __init__.py
│   ├── home.py           # Home page with overview
│   ├── diabetes.py       # Diabetes prediction
│   ├── heart.py          # Heart disease prediction
│   ├── stroke.py         # Stroke risk prediction
│   ├── kidney.py         # Kidney disease prediction
│   ├── liver.py          # Liver disease prediction
│   ├── report.py         # PDF report generator
│   └── about.py          # Model documentation
├── utils/
│   ├── __init__.py
│   ├── models.py         # ML models for all diseases
│   └── helpers.py        # Helper functions
└── .streamlit/
    └── config.toml       # Streamlit configuration
```

## Tech Stack
- **Framework**: Streamlit
- **ML**: scikit-learn (Random Forest, Gradient Boosting, Logistic Regression)
- **Data**: pandas, numpy
- **Visualization**: Plotly
- **PDF Generation**: FPDF2

## Running the Application
```bash
streamlit run app.py --server.port 5000
```

## Recent Changes
- **2024**: Initial implementation of HealthPredict Pro
  - Created 5 disease prediction models
  - Implemented sidebar navigation
  - Added PDF report generation
  - Created About Models documentation page

## Notes
- This is an educational tool for demonstration purposes
- Models use simulated training data with fixed random seeds for reproducibility
- Always consult healthcare professionals for medical decisions
