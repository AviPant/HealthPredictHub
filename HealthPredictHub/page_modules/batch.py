import streamlit as st
import pandas as pd
import numpy as np
import io
from utils.models import diabetes_model, heart_model, stroke_model, kidney_model, liver_model
from utils.database import save_batch_prediction, get_batch_predictions
import plotly.express as px
import plotly.graph_objects as go

def get_risk_level(probability):
    if probability >= 0.7:
        return 'High'
    elif probability >= 0.4:
        return 'Medium'
    return 'Low'

def render():
    st.markdown('<h1 class="main-header">📦 Batch Prediction</h1>', unsafe_allow_html=True)
    
    st.info("Upload patient data files to perform predictions on multiple records at once.")
    
    st.markdown("---")
    
    st.markdown("### 📁 Upload Patient Data")
    
    disease_type = st.selectbox(
        "Select Disease Type",
        options=["diabetes", "heart", "stroke", "kidney", "liver"],
        format_func=lambda x: x.title()
    )
    
    feature_templates = {
        'diabetes': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
        'heart': ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'],
        'stroke': ['Age', 'Hypertension', 'HeartDisease', 'AvgGlucose', 'BMI', 'Gender', 'EverMarried', 'WorkType', 'ResidenceType', 'SmokingStatus'],
        'kidney': ['Age', 'BloodPressure', 'SpecificGravity', 'Albumin', 'Sugar', 'BloodGlucoseRandom', 'BloodUrea', 'SerumCreatinine', 'Sodium', 'Potassium', 'Hemoglobin', 'PackedCellVolume', 'WhiteBloodCellCount', 'RedBloodCellCount'],
        'liver': ['Age', 'Gender', 'TotalBilirubin', 'DirectBilirubin', 'AlkalinePhosphatase', 'AlamineAminotransferase', 'AspartateAminotransferase', 'TotalProteins', 'Albumin', 'AlbuminGlobulinRatio']
    }
    
    with st.expander("📋 View Required Columns"):
        cols = feature_templates[disease_type]
        st.markdown("Your CSV file should contain these columns:")
        for i, col in enumerate(cols, 1):
            st.markdown(f"{i}. `{col}`")
    
    st.markdown("### 📥 Download Template")
    
    template_df = pd.DataFrame(columns=feature_templates[disease_type])
    template_df.loc[0] = [0] * len(feature_templates[disease_type])
    
    csv_template = template_df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV Template",
        data=csv_template,
        file_name=f"{disease_type}_template.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=['csv'],
        help="Upload a CSV file with patient data"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.markdown("### 📊 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            st.markdown(f"**Total Records:** {len(df)}")
            
            required_cols = feature_templates[disease_type]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                return
            
            st.markdown("---")
            
            batch_name = st.text_input("Batch Name", value=f"Batch_{disease_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}")
            
            if st.button("🔍 Run Batch Prediction", use_container_width=True):
                with st.spinner("Processing predictions..."):
                    models = {
                        'diabetes': diabetes_model,
                        'heart': heart_model,
                        'stroke': stroke_model,
                        'kidney': kidney_model,
                        'liver': liver_model
                    }
                    model = models[disease_type]
                    
                    results = []
                    progress_bar = st.progress(0)
                    
                    for idx, row in df.iterrows():
                        features = [row[col] for col in required_cols]
                        prediction, probability = model.predict(features)
                        risk_level = get_risk_level(probability)
                        
                        results.append({
                            'record_id': idx + 1,
                            'prediction': int(prediction),
                            'probability': float(probability),
                            'risk_level': risk_level,
                            'features': {col: float(row[col]) for col in required_cols}
                        })
                        
                        progress_bar.progress((idx + 1) / len(df))
                    
                    session_id = st.session_state.get('session_id', 'default')
                    batch_id = save_batch_prediction(batch_name, disease_type, results, session_id)
                    
                    st.success(f"✅ Batch prediction completed! Processed {len(df)} records.")
                    
                    st.markdown("### 📊 Results Summary")
                    
                    high_risk = sum(1 for r in results if r['risk_level'] == 'High')
                    medium_risk = sum(1 for r in results if r['risk_level'] == 'Medium')
                    low_risk = sum(1 for r in results if r['risk_level'] == 'Low')
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Records", len(results))
                    with col2:
                        st.metric("High Risk", high_risk, delta=None)
                    with col3:
                        st.metric("Medium Risk", medium_risk, delta=None)
                    with col4:
                        st.metric("Low Risk", low_risk, delta=None)
                    
                    summary_df = pd.DataFrame([
                        {'Risk Level': 'High', 'Count': high_risk, 'Percentage': f"{high_risk/len(results)*100:.1f}%"},
                        {'Risk Level': 'Medium', 'Count': medium_risk, 'Percentage': f"{medium_risk/len(results)*100:.1f}%"},
                        {'Risk Level': 'Low', 'Count': low_risk, 'Percentage': f"{low_risk/len(results)*100:.1f}%"}
                    ])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        colors = {'High': '#F44336', 'Medium': '#FFC107', 'Low': '#4CAF50'}
                        fig = px.pie(summary_df, values='Count', names='Risk Level',
                                    title='Risk Distribution',
                                    color='Risk Level',
                                    color_discrete_map=colors)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        prob_data = [r['probability'] * 100 for r in results]
                        fig = px.histogram(x=prob_data, nbins=20,
                                          title='Risk Score Distribution',
                                          labels={'x': 'Risk Score (%)', 'y': 'Count'})
                        fig.add_vline(x=70, line_dash="dash", line_color="red")
                        fig.add_vline(x=40, line_dash="dash", line_color="orange")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### 📋 Detailed Results")
                    
                    results_df = pd.DataFrame([{
                        'Record': r['record_id'],
                        'Risk Level': r['risk_level'],
                        'Risk Score': f"{r['probability']*100:.1f}%",
                        'Prediction': 'Positive' if r['prediction'] == 1 else 'Negative'
                    } for r in results])
                    
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    csv_results = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv_results,
                        file_name=f"{batch_name}_results.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    st.markdown("---")
    
    st.markdown("### 📜 Previous Batch Predictions")
    
    session_id = st.session_state.get('session_id', 'default')
    batch_history = get_batch_predictions(session_id=session_id, limit=10)
    
    if batch_history:
        for batch in batch_history:
            with st.expander(f"**{batch.batch_name}** - {batch.disease_type.title()} ({batch.created_at.strftime('%Y-%m-%d %H:%M')})"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total", batch.total_records)
                with col2:
                    st.metric("High Risk", batch.high_risk_count)
                with col3:
                    st.metric("Medium Risk", batch.medium_risk_count)
                with col4:
                    st.metric("Low Risk", batch.low_risk_count)
    else:
        st.info("No previous batch predictions found.")
