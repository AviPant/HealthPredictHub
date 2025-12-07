import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

HEALTHY_RANGES = {
    'diabetes': {
        'Glucose': {'min': 70, 'max': 100, 'unit': 'mg/dL', 'warning_max': 140},
        'Blood Pressure': {'min': 60, 'max': 80, 'unit': 'mmHg', 'warning_max': 90},
        'BMI': {'min': 18.5, 'max': 24.9, 'unit': 'kg/m²', 'warning_max': 30},
        'Insulin': {'min': 16, 'max': 166, 'unit': 'μU/ml', 'warning_max': 200},
        'Age': {'min': 21, 'max': 100, 'unit': 'years'}
    },
    'heart': {
        'Resting BP': {'min': 90, 'max': 120, 'unit': 'mmHg', 'warning_max': 140},
        'Cholesterol': {'min': 125, 'max': 200, 'unit': 'mg/dL', 'warning_max': 240},
        'Max Heart Rate': {'min': 60, 'max': 100, 'unit': 'bpm', 'warning_max': 150},
        'ST Depression': {'min': 0, 'max': 2, 'unit': '', 'warning_max': 4}
    },
    'stroke': {
        'Average Glucose': {'min': 70, 'max': 100, 'unit': 'mg/dL', 'warning_max': 150},
        'BMI': {'min': 18.5, 'max': 24.9, 'unit': 'kg/m²', 'warning_max': 30}
    },
    'kidney': {
        'Blood Pressure': {'min': 60, 'max': 80, 'unit': 'mmHg', 'warning_max': 100},
        'Serum Creatinine': {'min': 0.7, 'max': 1.3, 'unit': 'mg/dL', 'warning_max': 4},
        'Blood Urea': {'min': 7, 'max': 20, 'unit': 'mg/dL', 'warning_max': 80},
        'Hemoglobin': {'min': 12, 'max': 17, 'unit': 'g/dL', 'warning_min': 10},
        'Sodium': {'min': 136, 'max': 145, 'unit': 'mEq/L'},
        'Potassium': {'min': 3.5, 'max': 5.0, 'unit': 'mEq/L'}
    },
    'liver': {
        'Total Bilirubin': {'min': 0.1, 'max': 1.2, 'unit': 'mg/dL', 'warning_max': 3},
        'ALT': {'min': 7, 'max': 56, 'unit': 'IU/L', 'warning_max': 100},
        'AST': {'min': 10, 'max': 40, 'unit': 'IU/L', 'warning_max': 100},
        'Alkaline Phosphatase': {'min': 44, 'max': 147, 'unit': 'IU/L', 'warning_max': 350},
        'Albumin': {'min': 3.5, 'max': 5.0, 'unit': 'g/dL', 'warning_min': 3.0}
    }
}

def create_gauge_chart(value, min_val, max_val, title, unit, warning_max=None, warning_min=None):
    if warning_max:
        steps = [
            {'range': [min_val * 0.5, min_val], 'color': '#FFF9C4'},
            {'range': [min_val, max_val], 'color': '#C8E6C9'},
            {'range': [max_val, warning_max], 'color': '#FFF9C4'},
            {'range': [warning_max, warning_max * 1.5], 'color': '#FFCDD2'}
        ]
        range_max = warning_max * 1.5
    elif warning_min:
        steps = [
            {'range': [warning_min * 0.5, warning_min], 'color': '#FFCDD2'},
            {'range': [warning_min, min_val], 'color': '#FFF9C4'},
            {'range': [min_val, max_val], 'color': '#C8E6C9'},
            {'range': [max_val, max_val * 1.5], 'color': '#FFF9C4'}
        ]
        range_max = max_val * 1.5
    else:
        steps = [
            {'range': [min_val * 0.5, min_val], 'color': '#FFF9C4'},
            {'range': [min_val, max_val], 'color': '#C8E6C9'},
            {'range': [max_val, max_val * 1.5], 'color': '#FFF9C4'}
        ]
        range_max = max_val * 1.5
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': f"{title} ({unit})"},
        gauge={
            'axis': {'range': [min_val * 0.5, range_max]},
            'bar': {'color': '#1E88E5'},
            'steps': steps,
            'threshold': {
                'line': {'color': 'black', 'width': 2},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_comparison_bar(features, healthy_ranges):
    categories = []
    user_values = []
    healthy_mins = []
    healthy_maxs = []
    statuses = []
    
    for feature, value in features.items():
        if feature in healthy_ranges:
            hr = healthy_ranges[feature]
            categories.append(feature)
            
            normalized_value = (value - hr['min']) / (hr['max'] - hr['min']) * 100
            user_values.append(normalized_value)
            healthy_mins.append(0)
            healthy_maxs.append(100)
            
            if value < hr['min']:
                statuses.append('Low')
            elif value > hr['max']:
                if 'warning_max' in hr and value > hr['warning_max']:
                    statuses.append('High Risk')
                else:
                    statuses.append('Elevated')
            else:
                statuses.append('Normal')
    
    colors = {
        'Normal': '#4CAF50',
        'Elevated': '#FFC107',
        'Low': '#2196F3',
        'High Risk': '#F44336'
    }
    
    bar_colors = [colors.get(s, '#1E88E5') for s in statuses]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=categories,
        y=[100] * len(categories),
        name='Healthy Range',
        marker_color='rgba(200, 230, 201, 0.5)',
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Bar(
        x=categories,
        y=user_values,
        name='Your Values',
        marker_color=bar_colors,
        text=[f"{s}" for s in statuses],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Your Values vs Healthy Ranges',
        barmode='overlay',
        yaxis_title='Relative to Healthy Range (%)',
        xaxis_title='Health Metrics',
        height=400,
        showlegend=True
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=100, line_dash="dash", line_color="gray")
    
    return fig


def render():
    st.markdown('<h1 class="main-header">📊 Health Visualizations</h1>', unsafe_allow_html=True)
    
    st.info("Compare your health metrics against healthy reference ranges with interactive visualizations.")
    
    st.markdown("---")
    
    predictions = st.session_state.get('predictions', {})
    valid_predictions = {k: v for k, v in predictions.items() if v is not None}
    
    if not valid_predictions:
        st.warning("⚠️ No predictions available for visualization!")
        st.markdown("""
        To see visualizations, please complete at least one health risk assessment:
        
        1. 🩸 **Diabetes Predictor** - Assess diabetes risk
        2. ❤️ **Heart Disease Predictor** - Evaluate cardiovascular health
        3. 🧠 **Stroke Risk Predictor** - Check stroke risk factors
        4. 🫘 **Kidney Disease Predictor** - Analyze kidney function
        5. 🫁 **Liver Disease Predictor** - Evaluate liver health
        
        👈 Use the sidebar navigation to make predictions.
        """)
        return
    
    st.markdown("### 🎯 Select Assessment to Visualize")
    
    disease_options = list(valid_predictions.keys())
    disease_names = {
        'diabetes': '🩸 Diabetes',
        'heart': '❤️ Heart Disease',
        'stroke': '🧠 Stroke',
        'kidney': '🫘 Kidney Disease',
        'liver': '🫁 Liver Disease'
    }
    
    selected_disease = st.selectbox(
        "Choose Assessment",
        options=disease_options,
        format_func=lambda x: disease_names.get(x, x.title())
    )
    
    pred_data = valid_predictions[selected_disease]
    features = pred_data['features']
    result = pred_data['result']
    
    st.markdown("---")
    
    st.markdown("### 📈 Risk Overview")
    
    risk_colors = {
        'High': '#F44336',
        'Medium': '#FFC107',
        'Low': '#4CAF50'
    }
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=result['probability'] * 100,
            number={'suffix': '%'},
            title={'text': f"{disease_names.get(selected_disease, selected_disease.title())} Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': risk_colors.get(result['risk_level'], '#1E88E5')},
                'steps': [
                    {'range': [0, 40], 'color': '#C8E6C9'},
                    {'range': [40, 70], 'color': '#FFF9C4'},
                    {'range': [70, 100], 'color': '#FFCDD2'}
                ],
                'threshold': {
                    'line': {'color': 'black', 'width': 4},
                    'thickness': 0.75,
                    'value': result['probability'] * 100
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### 📊 Metrics Comparison")
    
    if selected_disease in HEALTHY_RANGES:
        healthy_ranges = HEALTHY_RANGES[selected_disease]
        
        comparable_features = {k: v for k, v in features.items() 
                             if k in healthy_ranges and isinstance(v, (int, float))}
        
        if comparable_features:
            fig = create_comparison_bar(comparable_features, healthy_ranges)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            st.markdown("### 🎛️ Individual Metric Gauges")
            
            cols = st.columns(min(3, len(comparable_features)))
            
            for i, (feature, value) in enumerate(comparable_features.items()):
                hr = healthy_ranges[feature]
                with cols[i % 3]:
                    fig = create_gauge_chart(
                        value=value,
                        min_val=hr['min'],
                        max_val=hr['max'],
                        title=feature,
                        unit=hr['unit'],
                        warning_max=hr.get('warning_max'),
                        warning_min=hr.get('warning_min')
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### 📋 Detailed Metrics Table")
    
    metrics_data = []
    for feature, value in features.items():
        status = "N/A"
        healthy_range = "N/A"
        
        if selected_disease in HEALTHY_RANGES and feature in HEALTHY_RANGES[selected_disease]:
            hr = HEALTHY_RANGES[selected_disease][feature]
            healthy_range = f"{hr['min']} - {hr['max']} {hr['unit']}"
            
            if isinstance(value, (int, float)):
                if value < hr['min']:
                    status = "⬇️ Low"
                elif value > hr['max']:
                    if 'warning_max' in hr and value > hr['warning_max']:
                        status = "🔴 High Risk"
                    else:
                        status = "⬆️ Elevated"
                else:
                    status = "✅ Normal"
        
        metrics_data.append({
            'Metric': feature,
            'Your Value': value,
            'Healthy Range': healthy_range,
            'Status': status
        })
    
    df = pd.DataFrame(metrics_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.markdown("### 💡 Recommendations")
    
    recommendations = []
    if selected_disease in HEALTHY_RANGES:
        for feature, value in features.items():
            if feature in HEALTHY_RANGES[selected_disease]:
                hr = HEALTHY_RANGES[selected_disease][feature]
                if isinstance(value, (int, float)):
                    if value > hr['max']:
                        recommendations.append(f"- **{feature}** is elevated ({value} {hr['unit']}). Consider lifestyle modifications to bring it within the healthy range ({hr['min']}-{hr['max']} {hr['unit']}).")
                    elif value < hr['min']:
                        recommendations.append(f"- **{feature}** is below normal ({value} {hr['unit']}). Consult with a healthcare provider about this reading.")
    
    if recommendations:
        st.warning("Based on your metrics, consider the following:")
        for rec in recommendations:
            st.markdown(rec)
    else:
        st.success("Your metrics are within healthy ranges. Continue maintaining your healthy lifestyle!")
