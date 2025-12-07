import streamlit as st
import plotly.graph_objects as go
from utils.models import MODEL_INFO

def render():
    st.markdown('<h1 class="main-header">ℹ️ About Models</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This page provides detailed information about the machine learning models used in HealthPredict Pro,
    including their algorithms, training data, features, and performance metrics.
    """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Model Performance Comparison")
    
    models = list(MODEL_INFO.keys())
    accuracies = [MODEL_INFO[m]['accuracy'] * 100 for m in models]
    names = [MODEL_INFO[m]['name'].replace(' Prediction Model', '') for m in models]
    
    colors = ['#1E88E5', '#43A047', '#FB8C00', '#8E24AA', '#E53935']
    
    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=accuracies,
            marker_color=colors,
            text=[f'{acc:.1f}%' for acc in accuracies],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Model Accuracy Comparison',
        xaxis_title='Disease Prediction Model',
        yaxis_title='Accuracy (%)',
        yaxis_range=[0, 100],
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### 🔬 Detailed Model Information")
    
    for key, info in MODEL_INFO.items():
        with st.expander(f"**{info['name']}**", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Description:**")
                st.write(info['description'])
                
                st.markdown(f"**Training Dataset:** {info['dataset']}")
                
                st.markdown("**Algorithm Details:**")
                if info['algorithm'] == 'Random Forest Classifier':
                    st.markdown("""
                    - Ensemble learning method using multiple decision trees
                    - Reduces overfitting through bootstrap aggregation
                    - Handles non-linear relationships effectively
                    - Robust to outliers and noise in data
                    """)
                elif info['algorithm'] == 'Gradient Boosting Classifier':
                    st.markdown("""
                    - Sequential ensemble method building trees iteratively
                    - Each tree corrects errors from previous trees
                    - High predictive accuracy for complex patterns
                    - Effective for imbalanced datasets
                    """)
                elif info['algorithm'] == 'Logistic Regression':
                    st.markdown("""
                    - Linear model for binary classification
                    - Provides probabilistic predictions
                    - Interpretable feature coefficients
                    - Efficient and fast training
                    """)
            
            with col2:
                st.metric("Accuracy", f"{info['accuracy']*100:.1f}%")
                st.metric("Features", info['features'])
                
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=info['accuracy']*100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#1E88E5"},
                        'steps': [
                            {'range': [0, 50], 'color': "#FFCDD2"},
                            {'range': [50, 75], 'color': "#FFF9C4"},
                            {'range': [75, 100], 'color': "#C8E6C9"}
                        ]
                    }
                ))
                gauge_fig.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(gauge_fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### 📚 Feature Descriptions")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Diabetes", "Heart", "Stroke", "Kidney", "Liver"])
    
    with tab1:
        st.markdown("""
        | Feature | Description | Normal Range |
        |---------|-------------|--------------|
        | Pregnancies | Number of pregnancies | 0-17 |
        | Glucose | Plasma glucose concentration (mg/dL) | 70-100 |
        | Blood Pressure | Diastolic blood pressure (mmHg) | 60-80 |
        | Skin Thickness | Triceps skin fold thickness (mm) | 10-50 |
        | Insulin | 2-Hour serum insulin (μU/ml) | 16-166 |
        | BMI | Body Mass Index (kg/m²) | 18.5-24.9 |
        | Diabetes Pedigree | Hereditary risk score | 0.08-2.4 |
        | Age | Age in years | - |
        """)
    
    with tab2:
        st.markdown("""
        | Feature | Description | Normal Range |
        |---------|-------------|--------------|
        | Age | Age in years | - |
        | Sex | Biological sex (M/F) | - |
        | Chest Pain Type | Type of chest pain (4 categories) | - |
        | Resting BP | Resting blood pressure (mmHg) | <120 |
        | Cholesterol | Serum cholesterol (mg/dL) | <200 |
        | Fasting BS | Fasting blood sugar >120 mg/dL | No |
        | Resting ECG | ECG results (3 categories) | Normal |
        | Max HR | Maximum heart rate achieved | 60-100 |
        | Exercise Angina | Exercise-induced angina | No |
        | Oldpeak | ST depression | 0-2 |
        | ST Slope | Slope of ST segment | Upsloping |
        """)
    
    with tab3:
        st.markdown("""
        | Feature | Description | Normal Range |
        |---------|-------------|--------------|
        | Age | Age in years | - |
        | Gender | Male/Female | - |
        | Hypertension | History of high BP | No |
        | Heart Disease | History of heart disease | No |
        | Ever Married | Marital status | - |
        | Work Type | Type of employment | - |
        | Residence Type | Urban/Rural | - |
        | Avg Glucose | Average glucose level (mg/dL) | 70-100 |
        | BMI | Body Mass Index (kg/m²) | 18.5-24.9 |
        | Smoking Status | Current smoking status | Never |
        """)
    
    with tab4:
        st.markdown("""
        | Feature | Description | Normal Range |
        |---------|-------------|--------------|
        | Age | Age in years | - |
        | Blood Pressure | BP measurement (mmHg) | 80-120 |
        | Specific Gravity | Urine specific gravity | 1.005-1.025 |
        | Albumin | Urine albumin level | 0 |
        | Sugar | Urine sugar level | 0 |
        | Blood Glucose | Random glucose (mg/dL) | 70-140 |
        | Blood Urea | BUN level (mg/dL) | 7-20 |
        | Serum Creatinine | Creatinine (mg/dL) | 0.7-1.3 |
        | Sodium | Blood sodium (mEq/L) | 136-145 |
        | Potassium | Blood potassium (mEq/L) | 3.5-5.0 |
        | Hemoglobin | Hemoglobin (g/dL) | 12-17 |
        """)
    
    with tab5:
        st.markdown("""
        | Feature | Description | Normal Range |
        |---------|-------------|--------------|
        | Age | Age in years | - |
        | Gender | Male/Female | - |
        | Total Bilirubin | Total bilirubin (mg/dL) | 0.1-1.2 |
        | Direct Bilirubin | Direct bilirubin (mg/dL) | 0-0.3 |
        | Alkaline Phosphatase | ALP enzyme (IU/L) | 44-147 |
        | ALT | Alanine aminotransferase (IU/L) | 7-56 |
        | AST | Aspartate aminotransferase (IU/L) | 10-40 |
        | Total Proteins | Serum proteins (g/dL) | 6.0-8.3 |
        | Albumin | Serum albumin (g/dL) | 3.5-5.0 |
        | A/G Ratio | Albumin/Globulin ratio | 1.0-2.5 |
        """)
    
    st.markdown("---")
    
    st.markdown("### ⚠️ Limitations & Considerations")
    
    st.warning("""
    **Important Model Limitations:**
    
    1. **Training Data Bias**: Models are trained on specific datasets that may not represent all populations equally.
    
    2. **Feature Availability**: Real-world predictions require accurate input data from proper medical testing.
    
    3. **Model Simplification**: These models use simplified feature sets and may not capture all risk factors.
    
    4. **No Temporal Factors**: Models don't account for changes over time or treatment effects.
    
    5. **Not Clinical Grade**: These models are for educational purposes and are not validated for clinical use.
    
    6. **Regional Variations**: Health risk factors may vary by geographic region and ethnicity.
    """)
    
    st.markdown("---")
    
    st.markdown("### 🛠️ Technical Implementation")
    
    with st.expander("View Technical Details"):
        st.markdown("""
        **Framework & Libraries:**
        - **Streamlit**: Web application framework
        - **Scikit-learn**: Machine learning models
        - **NumPy/Pandas**: Data manipulation
        - **Plotly**: Interactive visualizations
        - **FPDF2**: PDF report generation
        
        **Model Training Pipeline:**
        1. Data preprocessing and normalization
        2. Feature scaling using StandardScaler
        3. Train-test split (80-20)
        4. Model training with hyperparameter tuning
        5. Cross-validation for performance evaluation
        
        **Data Privacy:**
        - No data is stored permanently
        - All predictions are session-based
        - No external API calls for predictions
        """)
