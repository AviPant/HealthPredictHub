import streamlit as st
from utils.models import heart_model, MODEL_INFO
from utils.helpers import display_prediction_result

def render():
    st.markdown('<h1 class="main-header">❤️ Heart Disease Predictor</h1>', unsafe_allow_html=True)
    
    info = MODEL_INFO['heart']
    st.info(f"**Model:** {info['algorithm']} | **Accuracy:** {info['accuracy']*100:.1f}% | **Features:** {info['features']}")
    
    st.markdown("---")
    
    st.markdown("### Enter Your Cardiovascular Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider(
            "Age (years)",
            min_value=28, max_value=77, value=50,
            help="Your age in years"
        )
        
        sex = st.selectbox(
            "Sex",
            options=["Male", "Female"],
            help="Biological sex"
        )
        sex_val = 1 if sex == "Male" else 0
        
        chest_pain = st.selectbox(
            "Chest Pain Type",
            options=["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
            help="Type of chest pain experienced"
        )
        chest_pain_val = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain)
        
        resting_bp = st.slider(
            "Resting Blood Pressure (mmHg)",
            min_value=80, max_value=200, value=130,
            help="Resting blood pressure"
        )
        
        cholesterol = st.slider(
            "Cholesterol (mg/dL)",
            min_value=100, max_value=600, value=240,
            help="Serum cholesterol level"
        )
        
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL",
            options=["No", "Yes"],
            help="Is fasting blood sugar greater than 120 mg/dL?"
        )
        fasting_bs_val = 1 if fasting_bs == "Yes" else 0
    
    with col2:
        resting_ecg = st.selectbox(
            "Resting ECG Results",
            options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
            help="Resting electrocardiogram results"
        )
        resting_ecg_val = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
        
        max_hr = st.slider(
            "Maximum Heart Rate Achieved",
            min_value=60, max_value=202, value=150,
            help="Maximum heart rate achieved during exercise"
        )
        
        exercise_angina = st.selectbox(
            "Exercise Induced Angina",
            options=["No", "Yes"],
            help="Do you experience angina during exercise?"
        )
        exercise_angina_val = 1 if exercise_angina == "Yes" else 0
        
        oldpeak = st.slider(
            "ST Depression (Oldpeak)",
            min_value=0.0, max_value=6.5, value=1.0, step=0.1,
            help="ST depression induced by exercise relative to rest"
        )
        
        st_slope = st.selectbox(
            "ST Slope",
            options=["Upsloping", "Flat", "Downsloping"],
            help="Slope of the peak exercise ST segment"
        )
        st_slope_val = ["Upsloping", "Flat", "Downsloping"].index(st_slope)
    
    st.markdown("---")
    
    if st.button("🔍 Predict Heart Disease Risk", use_container_width=True):
        features = [age, sex_val, chest_pain_val, resting_bp, cholesterol,
                   fasting_bs_val, resting_ecg_val, max_hr, exercise_angina_val, 
                   oldpeak, st_slope_val]
        
        prediction, probability = heart_model.predict(features)
        
        st.markdown("### Prediction Results")
        
        input_features = {
            'Age': age,
            'Sex': sex,
            'Chest Pain Type': chest_pain,
            'Resting BP': resting_bp,
            'Cholesterol': cholesterol,
            'Fasting Blood Sugar >120': fasting_bs,
            'Resting ECG': resting_ecg,
            'Max Heart Rate': max_hr,
            'Exercise Angina': exercise_angina,
            'ST Depression': oldpeak,
            'ST Slope': st_slope
        }
        
        result = display_prediction_result(prediction, probability, "Heart Disease",
                                          input_features=input_features, disease_key='heart')
        
        st.session_state.predictions['heart'] = {
            'result': result,
            'features': input_features
        }
        
        st.success("✅ Prediction saved! Visit the Report Generator to download a comprehensive PDF report.")
        
        st.markdown("---")
        st.markdown("### Understanding Your Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Key Risk Factors:**")
            risk_factors = []
            if cholesterol > 240:
                risk_factors.append("- High cholesterol (>240 mg/dL)")
            if resting_bp > 140:
                risk_factors.append("- Elevated blood pressure (>140 mmHg)")
            if age > 55:
                risk_factors.append("- Age over 55 years")
            if chest_pain_val >= 2:
                risk_factors.append("- Concerning chest pain pattern")
            if exercise_angina_val == 1:
                risk_factors.append("- Exercise-induced angina")
            
            if risk_factors:
                st.markdown("\n".join(risk_factors))
            else:
                st.markdown("No major risk factors identified")
        
        with col2:
            st.markdown("**Recommendations:**")
            if prediction == 1:
                st.markdown("""
                - Consult a cardiologist promptly
                - Consider stress testing
                - Monitor blood pressure daily
                - Reduce sodium and saturated fat intake
                - Regular cardiovascular exercise
                """)
            else:
                st.markdown("""
                - Maintain heart-healthy lifestyle
                - Regular cardiovascular check-ups
                - Continue balanced diet
                - Stay physically active
                - Monitor cholesterol levels
                """)
