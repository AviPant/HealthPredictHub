import streamlit as st
from utils.models import stroke_model, MODEL_INFO
from utils.helpers import display_prediction_result

def render():
    st.markdown('<h1 class="main-header">🧠 Stroke Risk Predictor</h1>', unsafe_allow_html=True)
    
    info = MODEL_INFO['stroke']
    st.info(f"**Model:** {info['algorithm']} | **Accuracy:** {info['accuracy']*100:.1f}% | **Features:** {info['features']}")
    
    st.markdown("---")
    
    st.markdown("### Enter Your Health Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider(
            "Age (years)",
            min_value=0, max_value=100, value=45,
            help="Your age in years"
        )
        
        gender = st.selectbox(
            "Gender",
            options=["Male", "Female"],
            help="Your gender"
        )
        gender_val = 1 if gender == "Male" else 0
        
        hypertension = st.selectbox(
            "Do you have Hypertension?",
            options=["No", "Yes"],
            help="Have you been diagnosed with high blood pressure?"
        )
        hypertension_val = 1 if hypertension == "Yes" else 0
        
        heart_disease = st.selectbox(
            "Do you have Heart Disease?",
            options=["No", "Yes"],
            help="Have you been diagnosed with any heart condition?"
        )
        heart_disease_val = 1 if heart_disease == "Yes" else 0
        
        ever_married = st.selectbox(
            "Ever Married",
            options=["No", "Yes"],
            help="Have you ever been married?"
        )
        ever_married_val = 1 if ever_married == "Yes" else 0
    
    with col2:
        work_type = st.selectbox(
            "Work Type",
            options=["Children", "Never Worked", "Private", "Self-employed", "Government"],
            help="Type of employment"
        )
        work_type_val = ["Children", "Never Worked", "Private", "Self-employed", "Government"].index(work_type)
        
        residence_type = st.selectbox(
            "Residence Type",
            options=["Rural", "Urban"],
            help="Type of residence area"
        )
        residence_val = 1 if residence_type == "Urban" else 0
        
        avg_glucose = st.slider(
            "Average Glucose Level (mg/dL)",
            min_value=55.0, max_value=280.0, value=106.0, step=0.1,
            help="Average blood glucose level"
        )
        
        bmi = st.slider(
            "BMI (kg/m²)",
            min_value=10.0, max_value=100.0, value=28.0, step=0.1,
            help="Body Mass Index"
        )
        
        smoking_status = st.selectbox(
            "Smoking Status",
            options=["Never Smoked", "Formerly Smoked", "Smokes", "Unknown"],
            help="Current smoking status"
        )
        smoking_val = ["Never Smoked", "Formerly Smoked", "Smokes", "Unknown"].index(smoking_status)
    
    st.markdown("---")
    
    if st.button("🔍 Predict Stroke Risk", use_container_width=True):
        features = [age, hypertension_val, heart_disease_val, avg_glucose, bmi,
                   gender_val, ever_married_val, work_type_val, residence_val, smoking_val]
        
        prediction, probability = stroke_model.predict(features)
        
        st.markdown("### Prediction Results")
        
        input_features = {
            'Age': age,
            'Gender': gender,
            'Hypertension': hypertension,
            'Heart Disease': heart_disease,
            'Ever Married': ever_married,
            'Work Type': work_type,
            'Residence Type': residence_type,
            'Average Glucose': avg_glucose,
            'BMI': bmi,
            'Smoking Status': smoking_status
        }
        
        result = display_prediction_result(prediction, probability, "Stroke",
                                          input_features=input_features, disease_key='stroke')
        
        st.session_state.predictions['stroke'] = {
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
            if age > 65:
                risk_factors.append("- Age over 65 years")
            if hypertension_val == 1:
                risk_factors.append("- History of hypertension")
            if heart_disease_val == 1:
                risk_factors.append("- Existing heart disease")
            if avg_glucose > 150:
                risk_factors.append("- High glucose levels (>150 mg/dL)")
            if bmi > 30:
                risk_factors.append("- Elevated BMI (>30 kg/m²)")
            if smoking_val == 2:
                risk_factors.append("- Active smoking")
            
            if risk_factors:
                st.markdown("\n".join(risk_factors))
            else:
                st.markdown("No major risk factors identified")
        
        with col2:
            st.markdown("**Recommendations:**")
            if prediction == 1:
                st.markdown("""
                - Seek immediate medical consultation
                - Monitor blood pressure regularly
                - Control blood glucose levels
                - Consider lifestyle modifications
                - Learn FAST stroke warning signs
                """)
            else:
                st.markdown("""
                - Maintain healthy blood pressure
                - Regular exercise routine
                - Balanced, low-sodium diet
                - Avoid or quit smoking
                - Regular health screenings
                """)
