import streamlit as st
from utils.models import kidney_model, MODEL_INFO
from utils.helpers import display_prediction_result

def render():
    st.markdown('<h1 class="main-header">🫘 Kidney Disease Predictor</h1>', unsafe_allow_html=True)
    
    info = MODEL_INFO['kidney']
    st.info(f"**Model:** {info['algorithm']} | **Accuracy:** {info['accuracy']*100:.1f}% | **Features:** {info['features']}")
    
    st.markdown("---")
    
    st.markdown("### Enter Your Lab Test Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider(
            "Age (years)",
            min_value=2, max_value=90, value=50,
            help="Your age in years"
        )
        
        blood_pressure = st.slider(
            "Blood Pressure (mmHg)",
            min_value=50, max_value=180, value=80,
            help="Blood pressure measurement"
        )
        
        specific_gravity = st.slider(
            "Specific Gravity",
            min_value=1.005, max_value=1.025, value=1.015, step=0.001,
            help="Urine specific gravity"
        )
        
        albumin = st.slider(
            "Albumin Level (0-5)",
            min_value=0, max_value=5, value=0,
            help="Albumin in urine (0=nil to 5=high)"
        )
        
        sugar = st.slider(
            "Sugar Level (0-5)",
            min_value=0, max_value=5, value=0,
            help="Sugar in urine (0=nil to 5=high)"
        )
        
        blood_glucose = st.slider(
            "Blood Glucose Random (mg/dL)",
            min_value=22, max_value=500, value=140,
            help="Random blood glucose level"
        )
        
        blood_urea = st.slider(
            "Blood Urea (mg/dL)",
            min_value=1, max_value=400, value=50,
            help="Blood urea nitrogen level"
        )
    
    with col2:
        serum_creatinine = st.slider(
            "Serum Creatinine (mg/dL)",
            min_value=0.4, max_value=80.0, value=1.2, step=0.1,
            help="Serum creatinine level"
        )
        
        sodium = st.slider(
            "Sodium (mEq/L)",
            min_value=4, max_value=165, value=140,
            help="Blood sodium level"
        )
        
        potassium = st.slider(
            "Potassium (mEq/L)",
            min_value=2.5, max_value=50.0, value=4.5, step=0.1,
            help="Blood potassium level"
        )
        
        hemoglobin = st.slider(
            "Hemoglobin (g/dL)",
            min_value=3.0, max_value=18.0, value=12.5, step=0.1,
            help="Hemoglobin level"
        )
        
        packed_cell_volume = st.slider(
            "Packed Cell Volume (%)",
            min_value=9, max_value=55, value=40,
            help="Packed cell volume (hematocrit)"
        )
        
        wbc_count = st.slider(
            "White Blood Cell Count (cells/μL)",
            min_value=2200, max_value=27000, value=8000,
            help="White blood cell count"
        )
        
        rbc_count = st.slider(
            "Red Blood Cell Count (millions/μL)",
            min_value=2.0, max_value=8.0, value=4.7, step=0.1,
            help="Red blood cell count"
        )
    
    st.markdown("---")
    
    if st.button("🔍 Predict Kidney Disease Risk", use_container_width=True):
        features = [age, blood_pressure, specific_gravity, albumin, sugar,
                   blood_glucose, blood_urea, serum_creatinine, sodium,
                   potassium, hemoglobin, packed_cell_volume, wbc_count, rbc_count]
        
        prediction, probability = kidney_model.predict(features)
        
        st.markdown("### Prediction Results")
        
        input_features = {
            'Age': age,
            'Blood Pressure': blood_pressure,
            'Specific Gravity': specific_gravity,
            'Albumin': albumin,
            'Sugar': sugar,
            'Blood Glucose': blood_glucose,
            'Blood Urea': blood_urea,
            'Serum Creatinine': serum_creatinine,
            'Sodium': sodium,
            'Potassium': potassium,
            'Hemoglobin': hemoglobin,
            'Packed Cell Volume': packed_cell_volume,
            'WBC Count': wbc_count,
            'RBC Count': rbc_count
        }
        
        result = display_prediction_result(prediction, probability, "Chronic Kidney Disease",
                                          input_features=input_features, disease_key='kidney')
        
        st.session_state.predictions['kidney'] = {
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
            if serum_creatinine > 1.5:
                risk_factors.append("- Elevated creatinine (>1.5 mg/dL)")
            if blood_urea > 80:
                risk_factors.append("- High blood urea (>80 mg/dL)")
            if albumin > 2:
                risk_factors.append("- Significant albumin in urine")
            if hemoglobin < 11:
                risk_factors.append("- Low hemoglobin (<11 g/dL)")
            if blood_pressure > 100:
                risk_factors.append("- High blood pressure")
            if specific_gravity < 1.010:
                risk_factors.append("- Abnormal specific gravity")
            
            if risk_factors:
                st.markdown("\n".join(risk_factors))
            else:
                st.markdown("No major risk factors identified")
        
        with col2:
            st.markdown("**Recommendations:**")
            if prediction == 1:
                st.markdown("""
                - Consult a nephrologist
                - Monitor kidney function regularly
                - Control blood pressure
                - Manage blood glucose
                - Limit protein intake if advised
                - Stay well hydrated
                """)
            else:
                st.markdown("""
                - Maintain healthy lifestyle
                - Regular kidney function tests
                - Stay hydrated
                - Control blood pressure
                - Limit excessive salt intake
                """)
