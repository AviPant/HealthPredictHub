import streamlit as st
from utils.models import MODEL_INFO

def render():
    st.markdown('<h1 class="main-header">🏥 HealthPredict Pro</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; font-size: 1.2rem; color: #666;">
        Advanced Machine Learning Platform for Disease Risk Prediction
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🩸 Diabetes", f"{MODEL_INFO['diabetes']['accuracy']*100:.0f}%", "Accuracy")
    with col2:
        st.metric("❤️ Heart", f"{MODEL_INFO['heart']['accuracy']*100:.0f}%", "Accuracy")
    with col3:
        st.metric("🧠 Stroke", f"{MODEL_INFO['stroke']['accuracy']*100:.0f}%", "Accuracy")
    with col4:
        st.metric("🫘 Kidney", f"{MODEL_INFO['kidney']['accuracy']*100:.0f}%", "Accuracy")
    with col5:
        st.metric("🫁 Liver", f"{MODEL_INFO['liver']['accuracy']*100:.0f}%", "Accuracy")
    
    st.markdown("---")
    
    st.markdown("### 🎯 How It Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 1️⃣ Select Predictor
        Choose from 5 disease prediction models:
        - Diabetes
        - Heart Disease
        - Stroke
        - Kidney Disease
        - Liver Disease
        """)
    
    with col2:
        st.markdown("""
        #### 2️⃣ Enter Health Data
        Input your health metrics:
        - Blood pressure
        - Glucose levels
        - BMI and more
        - Lab test results
        """)
    
    with col3:
        st.markdown("""
        #### 3️⃣ Get Results
        Receive instant predictions:
        - Risk percentage
        - Risk level (Low/Medium/High)
        - Download PDF reports
        """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Available Predictors")
    
    for key, info in MODEL_INFO.items():
        with st.expander(f"**{info['name']}** - {info['algorithm']}"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Description:** {info['description']}")
                st.markdown(f"**Training Dataset:** {info['dataset']}")
            with col2:
                st.markdown(f"**Accuracy:** {info['accuracy']*100:.1f}%")
                st.markdown(f"**Features:** {info['features']}")
    
    st.markdown("---")
    
    st.markdown("### ⚠️ Important Disclaimer")
    st.warning("""
    **This tool is for educational and informational purposes only.**
    
    - Predictions are based on machine learning models and may not be 100% accurate
    - Results should NOT be used as a substitute for professional medical advice
    - Always consult qualified healthcare professionals for medical decisions
    - This platform does not store or share your health data
    """)
    
    st.markdown("---")
    
    st.markdown("### 🚀 Get Started")
    st.info("👈 Use the sidebar navigation to select a disease predictor and begin your health risk assessment.")
