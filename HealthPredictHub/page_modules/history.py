import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from utils.database import get_prediction_history, get_prediction_stats, get_all_predictions

def render():
    st.markdown('<h1 class="main-header">📜 Prediction History</h1>', unsafe_allow_html=True)
    
    st.info("View your prediction history and track health trends over time.")
    
    st.markdown("---")
    
    stats = get_prediction_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Predictions", stats['total'])
    with col2:
        high_risk = stats['by_risk'].get('High', 0)
        st.metric("High Risk", high_risk)
    with col3:
        medium_risk = stats['by_risk'].get('Medium', 0)
        st.metric("Medium Risk", medium_risk)
    with col4:
        low_risk = stats['by_risk'].get('Low', 0)
        st.metric("Low Risk", low_risk)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        disease_filter = st.selectbox(
            "Filter by Disease",
            options=["All", "diabetes", "heart", "stroke", "kidney", "liver"],
            format_func=lambda x: x.title() if x != "All" else x
        )
    
    with col2:
        limit = st.selectbox(
            "Show Records",
            options=[25, 50, 100, 200],
            index=1
        )
    
    disease_type = None if disease_filter == "All" else disease_filter
    predictions = get_prediction_history(disease_type=disease_type, limit=limit)
    
    if not predictions:
        st.warning("No prediction history found. Start making predictions to see your history here!")
        
        st.markdown("### 🚀 Quick Links")
        st.markdown("""
        - 🩸 **Diabetes Predictor** - Assess diabetes risk
        - ❤️ **Heart Disease Predictor** - Evaluate cardiovascular health
        - 🧠 **Stroke Risk Predictor** - Check stroke risk factors
        - 🫘 **Kidney Disease Predictor** - Analyze kidney function
        - 🫁 **Liver Disease Predictor** - Evaluate liver health
        
        👈 Use the sidebar navigation to make predictions.
        """)
        return
    
    st.markdown("### 📊 Prediction Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if stats['by_disease']:
            disease_df = pd.DataFrame([
                {'Disease': k.title(), 'Count': v} 
                for k, v in stats['by_disease'].items()
            ])
            fig = px.pie(disease_df, values='Count', names='Disease', 
                        title='Predictions by Disease Type',
                        color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if stats['by_risk']:
            risk_df = pd.DataFrame([
                {'Risk Level': k, 'Count': v} 
                for k, v in stats['by_risk'].items()
            ])
            colors = {'High': '#F44336', 'Medium': '#FFC107', 'Low': '#4CAF50'}
            fig = px.bar(risk_df, x='Risk Level', y='Count',
                        title='Predictions by Risk Level',
                        color='Risk Level',
                        color_discrete_map=colors)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### 📋 Prediction Records")
    
    records_data = []
    for pred in predictions:
        records_data.append({
            'Date': pred.created_at.strftime('%Y-%m-%d %H:%M'),
            'Disease': pred.disease_type.title(),
            'Risk Level': pred.risk_level,
            'Risk Score': f"{pred.probability * 100:.1f}%",
            'Result': 'Positive' if pred.prediction == 1 else 'Negative'
        })
    
    df = pd.DataFrame(records_data)
    
    def color_risk(val):
        colors = {
            'High': 'background-color: #FFCDD2',
            'Medium': 'background-color: #FFF9C4',
            'Low': 'background-color: #C8E6C9'
        }
        return colors.get(val, '')
    
    styled_df = df.style.applymap(color_risk, subset=['Risk Level'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.markdown("### 📈 Risk Trend Over Time")
    
    if len(predictions) > 1:
        trend_data = []
        for pred in reversed(predictions):
            trend_data.append({
                'Date': pred.created_at,
                'Risk Score': pred.probability * 100,
                'Disease': pred.disease_type.title()
            })
        
        trend_df = pd.DataFrame(trend_data)
        
        fig = px.line(trend_df, x='Date', y='Risk Score', color='Disease',
                     title='Risk Score Trend Over Time',
                     markers=True)
        fig.update_layout(yaxis_title='Risk Score (%)', xaxis_title='Date')
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     annotation_text="High Risk Threshold")
        fig.add_hline(y=40, line_dash="dash", line_color="orange",
                     annotation_text="Medium Risk Threshold")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Make more predictions to see your risk trends over time.")
    
    st.markdown("---")
    
    st.markdown("### 🔍 Detailed View")
    
    with st.expander("View Detailed Prediction Records"):
        for i, pred in enumerate(predictions[:10]):
            st.markdown(f"**{i+1}. {pred.disease_type.title()} Prediction** - {pred.created_at.strftime('%Y-%m-%d %H:%M')}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"Risk Level: **{pred.risk_level}**")
            with col2:
                st.markdown(f"Risk Score: **{pred.probability*100:.1f}%**")
            with col3:
                st.markdown(f"Result: **{'Positive' if pred.prediction == 1 else 'Negative'}**")
            
            if pred.input_features:
                with st.expander(f"View Input Features"):
                    for key, value in pred.input_features.items():
                        st.markdown(f"- {key}: {value}")
            
            st.markdown("---")
