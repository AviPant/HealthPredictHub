import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objects as go
import plotly.express as px
from utils.models import MODEL_INFO

def render():
    st.markdown('<h1 class="main-header">🔄 Model Retraining</h1>', unsafe_allow_html=True)
    
    st.info("Upload your own medical datasets to retrain prediction models and improve accuracy.")
    
    st.warning("""
    **⚠️ Important Notes:**
    - This feature is for educational and experimental purposes only
    - Retrained models are session-specific and not persisted
    - Always validate model performance before using for any predictions
    - Medical datasets should be properly anonymized and comply with privacy regulations
    """)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Select Model to Retrain")
    
    model_options = {
        'diabetes': 'Diabetes Prediction (Random Forest)',
        'heart': 'Heart Disease Prediction (Gradient Boosting)',
        'stroke': 'Stroke Risk Prediction (Random Forest)',
        'kidney': 'Kidney Disease Prediction (Gradient Boosting)',
        'liver': 'Liver Disease Prediction (Logistic Regression)'
    }
    
    selected_model = st.selectbox(
        "Choose Model",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x]
    )
    
    feature_requirements = {
        'diabetes': {
            'features': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
            'target': 'Outcome',
            'algorithm': RandomForestClassifier(n_estimators=100, random_state=42)
        },
        'heart': {
            'features': ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'],
            'target': 'HeartDisease',
            'algorithm': GradientBoostingClassifier(n_estimators=100, random_state=42)
        },
        'stroke': {
            'features': ['Age', 'Hypertension', 'HeartDisease', 'AvgGlucose', 'BMI', 'Gender', 'EverMarried', 'WorkType', 'ResidenceType', 'SmokingStatus'],
            'target': 'Stroke',
            'algorithm': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        },
        'kidney': {
            'features': ['Age', 'BloodPressure', 'SpecificGravity', 'Albumin', 'Sugar', 'BloodGlucoseRandom', 'BloodUrea', 'SerumCreatinine', 'Sodium', 'Potassium', 'Hemoglobin', 'PackedCellVolume', 'WhiteBloodCellCount', 'RedBloodCellCount'],
            'target': 'Classification',
            'algorithm': GradientBoostingClassifier(n_estimators=100, random_state=42)
        },
        'liver': {
            'features': ['Age', 'Gender', 'TotalBilirubin', 'DirectBilirubin', 'AlkalinePhosphatase', 'AlamineAminotransferase', 'AspartateAminotransferase', 'TotalProteins', 'Albumin', 'AlbuminGlobulinRatio'],
            'target': 'Dataset',
            'algorithm': LogisticRegression(random_state=42, max_iter=1000)
        }
    }
    
    requirements = feature_requirements[selected_model]
    
    st.markdown("---")
    
    st.markdown("### 📋 Dataset Requirements")
    
    with st.expander("View Required Columns"):
        st.markdown("**Feature Columns:**")
        for i, col in enumerate(requirements['features'], 1):
            st.markdown(f"{i}. `{col}`")
        st.markdown(f"\n**Target Column:** `{requirements['target']}` (0 = Negative, 1 = Positive)")
    
    st.markdown("### 📥 Download Template")
    
    all_cols = requirements['features'] + [requirements['target']]
    template_df = pd.DataFrame(columns=all_cols)
    template_df.loc[0] = [0] * len(all_cols)
    
    csv_template = template_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Training Data Template",
        data=csv_template,
        file_name=f"{selected_model}_training_template.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    st.markdown("### 📁 Upload Training Data")
    
    uploaded_file = st.file_uploader(
        "Upload CSV File with Training Data",
        type=['csv'],
        help="Upload a CSV file with features and target column"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.markdown("### 📊 Data Overview")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Total Features", len(df.columns) - 1)
            with col3:
                if requirements['target'] in df.columns:
                    positive = df[requirements['target']].sum()
                    st.metric("Positive Cases", f"{positive} ({positive/len(df)*100:.1f}%)")
            
            st.dataframe(df.head(10), use_container_width=True)
            
            missing_cols = [col for col in requirements['features'] + [requirements['target']] 
                          if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                return
            
            if df.isnull().any().any():
                st.warning("Dataset contains missing values. They will be handled during training.")
            
            st.markdown("---")
            
            st.markdown("### ⚙️ Training Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                test_size = st.slider("Test Set Size (%)", min_value=10, max_value=40, value=20)
                cv_folds = st.slider("Cross-Validation Folds", min_value=3, max_value=10, value=5)
            
            with col2:
                st.markdown("**Algorithm:** " + model_options[selected_model].split('(')[1].replace(')', ''))
                st.markdown(f"**Current Accuracy:** {MODEL_INFO[selected_model]['accuracy']*100:.1f}%")
            
            if st.button("🚀 Train Model", use_container_width=True):
                with st.spinner("Training model... This may take a moment."):
                    X = df[requirements['features']].copy()
                    y = df[requirements['target']].copy()
                    
                    X = X.fillna(X.mean())
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size/100, random_state=42
                    )
                    
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    model = requirements['algorithm']
                    model.fit(X_train_scaled, y_train)
                    
                    y_pred = model.predict(X_test_scaled)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    cv_scores = cross_val_score(model, scaler.fit_transform(X), y, cv=cv_folds)
                    
                    st.success("✅ Model training completed!")
                    
                    st.markdown("### 📊 Training Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        delta = (accuracy - MODEL_INFO[selected_model]['accuracy']) * 100
                        st.metric("Accuracy", f"{accuracy*100:.1f}%", 
                                 delta=f"{delta:+.1f}%" if abs(delta) > 0.1 else None)
                    with col2:
                        st.metric("Precision", f"{precision*100:.1f}%")
                    with col3:
                        st.metric("Recall", f"{recall*100:.1f}%")
                    with col4:
                        st.metric("F1 Score", f"{f1*100:.1f}%")
                    
                    st.markdown("### 📈 Cross-Validation Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        cv_df = pd.DataFrame({
                            'Fold': [f'Fold {i+1}' for i in range(cv_folds)],
                            'Accuracy': cv_scores * 100
                        })
                        fig = px.bar(cv_df, x='Fold', y='Accuracy',
                                    title=f'{cv_folds}-Fold Cross-Validation Scores',
                                    color='Accuracy',
                                    color_continuous_scale='Greens')
                        fig.add_hline(y=cv_scores.mean()*100, line_dash="dash",
                                     annotation_text=f"Mean: {cv_scores.mean()*100:.1f}%")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Cross-Validation Statistics:**")
                        st.markdown(f"- Mean Accuracy: **{cv_scores.mean()*100:.2f}%**")
                        st.markdown(f"- Std Deviation: **{cv_scores.std()*100:.2f}%**")
                        st.markdown(f"- Min Accuracy: **{cv_scores.min()*100:.2f}%**")
                        st.markdown(f"- Max Accuracy: **{cv_scores.max()*100:.2f}%**")
                    
                    if hasattr(model, 'feature_importances_'):
                        st.markdown("### 🎯 Feature Importance")
                        
                        importance_df = pd.DataFrame({
                            'Feature': requirements['features'],
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=True)
                        
                        fig = px.bar(importance_df, x='Importance', y='Feature',
                                    orientation='h',
                                    title='Feature Importance Ranking',
                                    color='Importance',
                                    color_continuous_scale='Blues')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### 📊 Model Comparison")
                    
                    comparison_df = pd.DataFrame({
                        'Metric': ['Accuracy', 'Training Data Size', 'Features Used'],
                        'Original Model': [
                            f"{MODEL_INFO[selected_model]['accuracy']*100:.1f}%",
                            MODEL_INFO[selected_model]['dataset'].split('(')[1].replace(' samples)', ''),
                            str(MODEL_INFO[selected_model]['features'])
                        ],
                        'Retrained Model': [
                            f"{accuracy*100:.1f}%",
                            str(len(df)),
                            str(len(requirements['features']))
                        ]
                    })
                    
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    st.info("""
                    **Note:** The retrained model is stored in session memory only and will be reset 
                    when the session ends. To use this model for predictions, the application would 
                    need to be updated to persist and load custom models.
                    """)
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)
    
    st.markdown("---")
    
    st.markdown("### 📚 Model Information")
    
    info = MODEL_INFO[selected_model]
    
    with st.expander("View Current Model Details"):
        st.markdown(f"**Name:** {info['name']}")
        st.markdown(f"**Algorithm:** {info['algorithm']}")
        st.markdown(f"**Current Accuracy:** {info['accuracy']*100:.1f}%")
        st.markdown(f"**Number of Features:** {info['features']}")
        st.markdown(f"**Training Dataset:** {info['dataset']}")
        st.markdown(f"**Description:** {info['description']}")
