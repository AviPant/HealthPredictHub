import streamlit as st

st.set_page_config(
    page_title="HealthPredict Pro",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

from page_modules import home, diabetes, heart, stroke, kidney, liver, report, about, history, batch, visualizations, retrain
import uuid

if 'predictions' not in st.session_state:
    st.session_state.predictions = {}

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .prediction-high {
        background-color: #FFCDD2;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #F44336;
    }
    .prediction-low {
        background-color: #C8E6C9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    .prediction-medium {
        background-color: #FFF9C4;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FFC107;
    }
    .metric-card {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<p class="sidebar-header">🏥 HealthPredict Pro</p>', unsafe_allow_html=True)
st.sidebar.markdown("---")

pages = {
    "🏠 Home": home,
    "🩸 Diabetes Predictor": diabetes,
    "❤️ Heart Disease Predictor": heart,
    "🧠 Stroke Risk Predictor": stroke,
    "🫘 Kidney Disease Predictor": kidney,
    "🫁 Liver Disease Predictor": liver,
    "📦 Batch Prediction": batch,
    "📈 Visualizations": visualizations,
    "📜 History": history,
    "🔄 Model Retraining": retrain,
    "📊 Report Generator": report,
    "ℹ️ About Models": about
}

selected_page = st.sidebar.radio("Navigation", list(pages.keys()), label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Stats")
predictions_count = len([k for k, v in st.session_state.predictions.items() if v is not None])
st.sidebar.metric("Predictions Made", predictions_count)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    <p>⚠️ Medical Disclaimer</p>
    <p style='font-size: 0.7rem;'>This tool is for educational purposes only. 
    Always consult healthcare professionals for medical decisions.</p>
</div>
""", unsafe_allow_html=True)

try:
    pages[selected_page].render()
except Exception as e:
    st.error(f"Error loading page: {str(e)}")
    st.exception(e)
