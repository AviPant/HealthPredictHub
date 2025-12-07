import streamlit as st
from utils.models import diabetes_model, MODEL_INFO
from utils.helpers import display_prediction_result

def render():
    st.markdown('<h1 class="main-header">🩸 Diabetes Predictor</h1>', unsafe_allow_html=True)
    
    info = MODEL_INFO['diabetes']
    st.info(f"**Model:** {info['algorithm']} | **Accuracy:** {info['accuracy']*100:.1f}% | **Features:** {info['features']}")
    
    st.markdown("---")
    
    st.markdown("### Enter Your Health Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0, max_value=20, value=1,
            help="Number of times pregnant"
        )
        
        glucose = st.slider(
            "Glucose Level (mg/dL)",
            min_value=0, max_value=200, value=120,
            help="Plasma glucose concentration (2 hours after glucose tolerance test)"
        )
        
        blood_pressure = st.slider(
            "Blood Pressure (mmHg)",
            min_value=0, max_value=140, value=72,
            help="Diastolic blood pressure"
        )
        
        skin_thickness = st.slider(
            "Skin Thickness (mm)",
            min_value=0, max_value=100, value=29,
            help="Triceps skin fold thickness"
        )
    
    with col2:
        insulin = st.slider(
            "Insulin Level (μU/ml)",
            min_value=0, max_value=900, value=140,
            help="2-Hour serum insulin"
        )
        
        bmi = st.slider(
            "BMI (kg/m²)",
            min_value=0.0, max_value=70.0, value=32.0, step=0.1,
            help="Body Mass Index"
        )
        
        dpf = st.slider(
            "Diabetes Pedigree Function",
            min_value=0.0, max_value=2.5, value=0.47, step=0.01,
            help="Diabetes hereditary risk score"
        )
        
        age = st.slider(
            "Age (years)",
            min_value=21, max_value=100, value=33,
            help="Your age in years"
        )
    
    st.markdown("---")
    
    if st.button("🔍 Predict Diabetes Risk", use_container_width=True):
        features = [pregnancies, glucose, blood_pressure, skin_thickness, 
                   insulin, bmi, dpf, age]
        
        prediction, probability = diabetes_model.predict(features)
        
        st.markdown("### Prediction Results")
        
        input_features = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'Blood Pressure': blood_pressure,
            'Skin Thickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'Diabetes Pedigree Function': dpf,
            'Age': age
        }
        
        result = display_prediction_result(prediction, probability, "Diabetes", 
                                          input_features=input_features, disease_key='diabetes')
        
        st.session_state.predictions['diabetes'] = {
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
            if glucose > 140:
                risk_factors.append("- High glucose levels (>140 mg/dL)")
            if bmi > 30:
                risk_factors.append("- Elevated BMI (>30 kg/m²)")
            if age > 45:
                risk_factors.append("- Age over 45 years")
            if blood_pressure > 90:
                risk_factors.append("- High blood pressure (>90 mmHg)")
            if dpf > 0.5:
                risk_factors.append("- Family history of diabetes")
            
            if risk_factors:
                st.markdown("\n".join(risk_factors))
            else:
                st.markdown("No major risk factors identified")
        
        with col2:
            st.markdown("**Recommendations:**")
            if prediction == 1:
                st.markdown("""
                - Schedule a consultation with your doctor
                - Consider HbA1c testing
                - Monitor blood glucose regularly
                - Maintain a healthy diet and exercise
                """)
            else:
                st.markdown("""
                - Continue healthy lifestyle habits
                - Regular health check-ups
                - Maintain balanced diet
                - Stay physically active
                """)
