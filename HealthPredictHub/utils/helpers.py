import streamlit as st
from datetime import datetime
from utils.database import save_prediction

def display_prediction_result(prediction, probability, disease_name, input_features=None, disease_key=None):
    risk_percentage = probability * 100
    
    if risk_percentage >= 70:
        risk_level = "High"
        css_class = "prediction-high"
        color = "#F44336"
    elif risk_percentage >= 40:
        risk_level = "Medium"
        css_class = "prediction-medium"
        color = "#FFC107"
    else:
        risk_level = "Low"
        css_class = "prediction-low"
        color = "#4CAF50"
    
    st.markdown(f"""
    <div class="{css_class}">
        <h3 style="margin:0; color: {color};">{risk_level} Risk - {disease_name}</h3>
        <p style="font-size: 1.2rem; margin: 0.5rem 0;">Risk Score: {risk_percentage:.1f}%</p>
        <p style="margin: 0;">Prediction: {"Positive (At Risk)" if prediction == 1 else "Negative (Low Risk)"}</p>
    </div>
    """, unsafe_allow_html=True)
    
    result = {
        'disease': disease_name,
        'prediction': int(prediction),
        'probability': float(probability),
        'risk_level': risk_level,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if disease_key and input_features:
        session_id = st.session_state.get('session_id', 'default')
        save_prediction(
            disease_type=disease_key,
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level,
            input_features=input_features,
            session_id=session_id
        )
    
    return result


def create_metric_card(title, value, description=""):
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="margin:0; color: #1E88E5;">{title}</h4>
        <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{value}</p>
        <p style="font-size: 0.8rem; color: #666; margin: 0;">{description}</p>
    </div>
    """, unsafe_allow_html=True)


def get_risk_color(risk_level):
    colors = {
        'High': '#F44336',
        'Medium': '#FFC107',
        'Low': '#4CAF50'
    }
    return colors.get(risk_level, '#1E88E5')


def format_feature_value(value, feature_name):
    if 'pressure' in feature_name.lower():
        return f"{value:.0f} mmHg"
    elif 'glucose' in feature_name.lower():
        return f"{value:.0f} mg/dL"
    elif 'bmi' in feature_name.lower():
        return f"{value:.1f} kg/m²"
    elif 'cholesterol' in feature_name.lower():
        return f"{value:.0f} mg/dL"
    elif 'age' in feature_name.lower():
        return f"{value:.0f} years"
    else:
        return f"{value:.2f}"
