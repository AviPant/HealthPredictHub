import streamlit as st
from utils.models import liver_model, MODEL_INFO
from utils.helpers import display_prediction_result

def render():
    st.markdown('<h1 class="main-header">🫁 Liver Disease Predictor</h1>', unsafe_allow_html=True)
    
    info = MODEL_INFO['liver']
    st.info(f"**Model:** {info['algorithm']} | **Accuracy:** {info['accuracy']*100:.1f}% | **Features:** {info['features']}")
    
    st.markdown("---")
    
    st.markdown("### Enter Your Liver Function Test Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider(
            "Age (years)",
            min_value=4, max_value=90, value=45,
            help="Your age in years"
        )
        
        gender = st.selectbox(
            "Gender",
            options=["Male", "Female"],
            help="Your gender"
        )
        gender_val = 1 if gender == "Male" else 0
        
        total_bilirubin = st.slider(
            "Total Bilirubin (mg/dL)",
            min_value=0.4, max_value=75.0, value=1.0, step=0.1,
            help="Total bilirubin level"
        )
        
        direct_bilirubin = st.slider(
            "Direct Bilirubin (mg/dL)",
            min_value=0.1, max_value=20.0, value=0.3, step=0.1,
            help="Direct (conjugated) bilirubin level"
        )
        
        alkaline_phosphatase = st.slider(
            "Alkaline Phosphatase (IU/L)",
            min_value=63, max_value=2200, value=290,
            help="Alkaline phosphatase enzyme level"
        )
    
    with col2:
        alt = st.slider(
            "ALT - Alanine Aminotransferase (IU/L)",
            min_value=10, max_value=2000, value=50,
            help="ALT (SGPT) enzyme level"
        )
        
        ast = st.slider(
            "AST - Aspartate Aminotransferase (IU/L)",
            min_value=10, max_value=5000, value=60,
            help="AST (SGOT) enzyme level"
        )
        
        total_proteins = st.slider(
            "Total Proteins (g/dL)",
            min_value=2.7, max_value=10.0, value=6.5, step=0.1,
            help="Total serum proteins"
        )
        
        albumin = st.slider(
            "Albumin (g/dL)",
            min_value=0.9, max_value=5.5, value=3.1, step=0.1,
            help="Serum albumin level"
        )
        
        ag_ratio = st.slider(
            "Albumin/Globulin Ratio",
            min_value=0.3, max_value=3.0, value=0.95, step=0.01,
            help="Ratio of albumin to globulin"
        )
    
    st.markdown("---")
    
    if st.button("🔍 Predict Liver Disease Risk", use_container_width=True):
        features = [age, gender_val, total_bilirubin, direct_bilirubin,
                   alkaline_phosphatase, alt, ast, total_proteins, albumin, ag_ratio]
        
        prediction, probability = liver_model.predict(features)
        
        st.markdown("### Prediction Results")
        
        input_features = {
            'Age': age,
            'Gender': gender,
            'Total Bilirubin': total_bilirubin,
            'Direct Bilirubin': direct_bilirubin,
            'Alkaline Phosphatase': alkaline_phosphatase,
            'ALT': alt,
            'AST': ast,
            'Total Proteins': total_proteins,
            'Albumin': albumin,
            'A/G Ratio': ag_ratio
        }
        
        result = display_prediction_result(prediction, probability, "Liver Disease",
                                          input_features=input_features, disease_key='liver')
        
        st.session_state.predictions['liver'] = {
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
            if total_bilirubin > 1.2:
                risk_factors.append("- Elevated bilirubin (>1.2 mg/dL)")
            if alt > 56:
                risk_factors.append("- High ALT levels (>56 IU/L)")
            if ast > 40:
                risk_factors.append("- Elevated AST levels (>40 IU/L)")
            if alkaline_phosphatase > 350:
                risk_factors.append("- High alkaline phosphatase")
            if albumin < 3.5:
                risk_factors.append("- Low albumin (<3.5 g/dL)")
            if ag_ratio < 0.8:
                risk_factors.append("- Low A/G ratio")
            
            if risk_factors:
                st.markdown("\n".join(risk_factors))
            else:
                st.markdown("No major risk factors identified")
        
        with col2:
            st.markdown("**Recommendations:**")
            if prediction == 1:
                st.markdown("""
                - Consult a hepatologist
                - Consider liver ultrasound
                - Avoid alcohol completely
                - Review medications with doctor
                - Monitor liver enzymes regularly
                - Follow liver-friendly diet
                """)
            else:
                st.markdown("""
                - Maintain healthy lifestyle
                - Limit alcohol consumption
                - Regular liver function tests
                - Balanced nutrition
                - Maintain healthy weight
                """)
