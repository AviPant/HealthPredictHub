import streamlit as st
from fpdf import FPDF
from datetime import datetime
import io
import tempfile
import os
import plotly.graph_objects as go
import plotly.io as pio
from utils.models import MODEL_INFO
from utils.database import get_prediction_history as get_predictions


class HealthReportPDF(FPDF):

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font('Helvetica', 'B', 20)
        self.set_text_color(30, 136, 229)
        self.cell(0, 10, 'HealthPredict Pro', ln=True, align='C')
        self.set_font('Helvetica', '', 12)
        self.set_text_color(100, 100, 100)
        self.cell(0,
                  8,
                  'Comprehensive Health Risk Assessment Report',
                  ln=True,
                  align='C')
        self.ln(5)
        self.set_draw_color(30, 136, 229)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(
            0,
            10,
            f'Page {self.page_no()} | Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")} | For educational purposes only',
            align='C')

    def add_prediction_section(self, disease_name, prediction_data):
        result = prediction_data['result']
        features = prediction_data['features']

        risk_colors = {
            'High': (244, 67, 54),
            'Medium': (255, 193, 7),
            'Low': (76, 175, 80)
        }
        color = risk_colors.get(result['risk_level'], (30, 136, 229))

        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(30, 136, 229)
        self.cell(0, 10, disease_name, ln=True)

        self.set_fill_color(*color)
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 12)
        self.cell(60, 10, f"Risk Level: {result['risk_level']}", fill=True)
        self.cell(60,
                  10,
                  f"Risk Score: {result['probability']*100:.1f}%",
                  fill=True)
        status = "Positive" if result['prediction'] == 1 else "Negative"
        self.cell(70, 10, f"Prediction: {status}", fill=True, ln=True)

        self.ln(5)
        self.set_text_color(0, 0, 0)
        self.set_font('Helvetica', 'B', 11)
        self.cell(0, 8, "Input Parameters:", ln=True)

        self.set_font('Helvetica', '', 10)
        col_width = 95
        for i, (key, value) in enumerate(features.items()):
            if i % 2 == 0:
                self.cell(col_width, 6, f"{key}: {value}")
            else:
                self.cell(col_width, 6, f"{key}: {value}", ln=True)

        if len(features) % 2 != 0:
            self.ln()

        self.ln(5)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(8)

    def add_risk_overview_chart(self, predictions):
        disease_names = {
            'diabetes': 'Diabetes',
            'heart': 'Heart Disease',
            'stroke': 'Stroke',
            'kidney': 'Kidney Disease',
            'liver': 'Liver Disease'
        }

        names = []
        probabilities = []
        colors = []

        for disease_key, pred_data in predictions.items():
            if pred_data is not None:
                names.append(
                    disease_names.get(disease_key, disease_key.title()))
                probabilities.append(pred_data['result']['probability'] * 100)
                risk_level = pred_data['result']['risk_level']
                if risk_level == 'High':
                    colors.append('#F44336')
                elif risk_level == 'Medium':
                    colors.append('#FFC107')
                else:
                    colors.append('#4CAF50')

        if not names:
            return

        fig = go.Figure(data=[
            go.Bar(x=names,
                   y=probabilities,
                   marker_color=colors,
                   text=[f'{p:.1f}%' for p in probabilities],
                   textposition='outside')
        ])

        fig.update_layout(title='Risk Score Overview',
                          xaxis_title='Disease Type',
                          yaxis_title='Risk Score (%)',
                          yaxis_range=[0, 100],
                          template='plotly_white',
                          width=700,
                          height=400)

        with tempfile.NamedTemporaryFile(suffix='.png',
                                         delete=False) as tmp_file:
            pio.write_image(fig, tmp_file.name, format='png', scale=2)
            self.add_page()
            self.set_font('Helvetica', 'B', 16)
            self.set_text_color(30, 136, 229)
            self.cell(0, 10, 'Risk Overview Chart', ln=True)
            self.ln(5)
            self.image(tmp_file.name, x=15, w=180)
            os.unlink(tmp_file.name)

    def add_trend_analysis(self, session_id):
        db_predictions = get_predictions(session_id=session_id, limit=100)

        if not db_predictions:
            return

        disease_data = {}
        for pred in db_predictions:
            disease_type = pred.disease_type
            if disease_type not in disease_data:
                disease_data[disease_type] = {'dates': [], 'probabilities': []}
            disease_data[disease_type]['dates'].append(pred.created_at)
            disease_data[disease_type]['probabilities'].append(
                pred.probability * 100)

        if not disease_data:
            return

        fig = go.Figure()

        colors = {
            'diabetes': '#E91E63',
            'heart': '#F44336',
            'stroke': '#9C27B0',
            'kidney': '#3F51B5',
            'liver': '#009688'
        }

        disease_names = {
            'diabetes': 'Diabetes',
            'heart': 'Heart Disease',
            'stroke': 'Stroke',
            'kidney': 'Kidney Disease',
            'liver': 'Liver Disease'
        }

        for disease_type, data in disease_data.items():
            if len(data['dates']) > 1:
                fig.add_trace(
                    go.Scatter(
                        x=data['dates'],
                        y=data['probabilities'],
                        mode='lines+markers',
                        name=disease_names.get(disease_type,
                                               disease_type.title()),
                        line=dict(color=colors.get(disease_type, '#1E88E5'))))

        if not fig.data:
            return

        fig.update_layout(title='Risk Score Trends Over Time',
                          xaxis_title='Date',
                          yaxis_title='Risk Score (%)',
                          yaxis_range=[0, 100],
                          template='plotly_white',
                          width=700,
                          height=400,
                          legend=dict(orientation='h',
                                      yanchor='bottom',
                                      y=1.02,
                                      xanchor='right',
                                      x=1))

        with tempfile.NamedTemporaryFile(suffix='.png',
                                         delete=False) as tmp_file:
            pio.write_image(fig, tmp_file.name, format='png', scale=2)
            self.add_page()
            self.set_font('Helvetica', 'B', 16)
            self.set_text_color(30, 136, 229)
            self.cell(0, 10, 'Health Trend Analysis', ln=True)
            self.ln(5)
            self.image(tmp_file.name, x=15, w=180)
            os.unlink(tmp_file.name)

            self.ln(10)
            self.set_font('Helvetica', '', 11)
            self.set_text_color(0, 0, 0)
            self.multi_cell(
                0, 6,
                "This chart shows how your health risk scores have changed over time. Tracking trends can help identify improvements or areas that need attention."
            )


def generate_pdf_report(predictions,
                        include_charts=True,
                        include_trends=True,
                        session_id=None):
    pdf = HealthReportPDF()
    pdf.add_page()

    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(
        0,
        6,
        f"Report Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
        ln=True)
    pdf.cell(0, 6, f"Total Assessments: {len(predictions)}", ln=True)
    pdf.ln(10)

    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(30, 136, 229)
    pdf.cell(0, 10, "Risk Assessment Summary", ln=True)
    pdf.ln(5)

    disease_names = {
        'diabetes': 'Diabetes Risk Assessment',
        'heart': 'Heart Disease Risk Assessment',
        'stroke': 'Stroke Risk Assessment',
        'kidney': 'Kidney Disease Risk Assessment',
        'liver': 'Liver Disease Risk Assessment'
    }

    for disease_key, pred_data in predictions.items():
        if pred_data is not None:
            disease_name = disease_names.get(disease_key, disease_key.title())
            pdf.add_prediction_section(disease_name, pred_data)

    if include_charts:
        try:
            pdf.add_risk_overview_chart(predictions)
        except Exception as e:
            pass

    if include_trends and session_id:
        try:
            pdf.add_trend_analysis(session_id)
        except Exception as e:
            pass

    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(30, 136, 229)
    pdf.cell(0, 10, "Model Information", ln=True)
    pdf.ln(5)

    for key, info in MODEL_INFO.items():
        if key in predictions and predictions[key] is not None:
            pdf.set_font('Helvetica', 'B', 12)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 8, info['name'], ln=True)

            pdf.set_font('Helvetica', '', 10)
            pdf.cell(0, 6, f"Algorithm: {info['algorithm']}", ln=True)
            pdf.cell(0, 6, f"Accuracy: {info['accuracy']*100:.1f}%", ln=True)
            pdf.cell(0, 6, f"Features Used: {info['features']}", ln=True)
            pdf.multi_cell(0, 6, f"Description: {info['description']}")
            pdf.ln(5)

    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(244, 67, 54)
    pdf.cell(0, 10, "Important Disclaimer", ln=True)
    pdf.ln(5)

    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(0, 0, 0)
    disclaimer = """
This health risk assessment report is generated by HealthPredict Pro, an educational tool designed to demonstrate the application of machine learning in healthcare.

IMPORTANT NOTICES:

1. This tool is for EDUCATIONAL and INFORMATIONAL purposes only.

2. The predictions made by this system are based on machine learning models trained on historical datasets and may not accurately reflect your actual health status.

3. These results should NEVER be used as a substitute for professional medical advice, diagnosis, or treatment.

4. Always seek the advice of qualified healthcare professionals regarding any medical conditions or health concerns.

5. The accuracy metrics displayed represent performance on test datasets and may vary in real-world applications.

6. No personal health information is stored or transmitted by this application.

7. If you are experiencing a medical emergency, please contact emergency services immediately.

By using this tool, you acknowledge that you understand these limitations and agree to use the information responsibly.
    """
    pdf.multi_cell(0, 6, disclaimer.strip())

    pdf_bytes = pdf.output()
    # Convert bytearray to bytes if needed
    if isinstance(pdf_bytes, bytearray):
        return bytes(pdf_bytes)
    return pdf_bytes


def render():
    st.markdown('<h1 class="main-header">📊 Advanced Report Generator</h1>',
                unsafe_allow_html=True)

    st.info(
        "Generate comprehensive PDF reports with charts, trend analysis, and detailed health risk assessments."
    )

    st.markdown("---")

    predictions = st.session_state.get('predictions', {})
    valid_predictions = {k: v for k, v in predictions.items() if v is not None}
    session_id = st.session_state.get('session_id', 'default')

    if not valid_predictions:
        st.warning("⚠️ No predictions available yet!")
        st.markdown("""
        To generate a report, please complete at least one health risk assessment:
        
        1. 🩸 **Diabetes Predictor** - Assess diabetes risk
        2. ❤️ **Heart Disease Predictor** - Evaluate cardiovascular health
        3. 🧠 **Stroke Risk Predictor** - Check stroke risk factors
        4. 🫘 **Kidney Disease Predictor** - Analyze kidney function
        5. 🫁 **Liver Disease Predictor** - Evaluate liver health
        
        👈 Use the sidebar navigation to access these predictors.
        """)
        return

    st.markdown("### 📋 Available Assessments")

    disease_icons = {
        'diabetes': '🩸',
        'heart': '❤️',
        'stroke': '🧠',
        'kidney': '🫘',
        'liver': '🫁'
    }

    disease_names = {
        'diabetes': 'Diabetes',
        'heart': 'Heart Disease',
        'stroke': 'Stroke',
        'kidney': 'Kidney Disease',
        'liver': 'Liver Disease'
    }

    cols = st.columns(len(valid_predictions))
    for i, (disease, data) in enumerate(valid_predictions.items()):
        with cols[i]:
            icon = disease_icons.get(disease, '📊')
            name = disease_names.get(disease, disease.title())
            result = data['result']

            risk_color = {
                'High': '#F44336',
                'Medium': '#FFC107',
                'Low': '#4CAF50'
            }.get(result['risk_level'], '#1E88E5')

            st.markdown(f"""
            <div style="background-color: #f5f5f5; padding: 1rem; border-radius: 10px; text-align: center; border-left: 4px solid {risk_color};">
                <h3 style="margin: 0;">{icon} {name}</h3>
                <p style="font-size: 1.5rem; font-weight: bold; color: {risk_color}; margin: 0.5rem 0;">{result['risk_level']} Risk</p>
                <p style="margin: 0;">{result['probability']*100:.1f}%</p>
            </div>
            """,
                        unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📈 Report Options")

    col1, col2 = st.columns(2)
    with col1:
        include_charts = st.checkbox(
            "Include Risk Overview Charts",
            value=True,
            help="Add visual bar charts showing risk scores")
    with col2:
        include_trends = st.checkbox(
            "Include Trend Analysis",
            value=True,
            help="Add historical trend charts (requires saved predictions)")

    st.markdown("---")

    st.markdown("### 📥 Generate & Download Report")

    # Initialize session state for PDF if not exists
    if 'pdf_report_bytes' not in st.session_state:
        st.session_state.pdf_report_bytes = None
    if 'pdf_report_filename' not in st.session_state:
        st.session_state.pdf_report_filename = None

    if st.button("🔄 Generate PDF Report", use_container_width=True):
        with st.spinner(
                "Generating your comprehensive health report with charts..."):
            try:
                pdf_bytes = generate_pdf_report(valid_predictions,
                                                include_charts=include_charts,
                                                include_trends=include_trends,
                                                session_id=session_id)

                # Ensure it's bytes, not bytearray
                if isinstance(pdf_bytes, bytearray):
                    pdf_bytes = bytes(pdf_bytes)
                
                # Store in session state
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"HealthPredict_Report_{timestamp}.pdf"
                
                st.session_state.pdf_report_bytes = pdf_bytes
                st.session_state.pdf_report_filename = filename

                st.success("✅ Report generated successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
                st.exception(e)

    # Show download button if PDF is ready
    if st.session_state.pdf_report_bytes is not None:
        st.download_button(
            label="📥 Download PDF Report",
            data=st.session_state.pdf_report_bytes,
            file_name=st.session_state.pdf_report_filename,
            mime="application/pdf",
            use_container_width=True
        )

    st.markdown("---")

    db_predictions = get_predictions(session_id=session_id, limit=10)
    if db_predictions:
        st.markdown("### 📜 Recent Prediction History")
        st.markdown("*Your last 10 predictions from this session:*")

        for pred in db_predictions[:5]:
            disease_icon = disease_icons.get(pred.disease_type, '📊')
            disease_name = disease_names.get(pred.disease_type,
                                             pred.disease_type.title())
            risk_color = {
                'High': '#F44336',
                'Medium': '#FFC107',
                'Low': '#4CAF50'
            }.get(pred.risk_level, '#1E88E5')

            st.markdown(f"""
            <div style="background-color: #f9f9f9; padding: 0.5rem 1rem; border-radius: 5px; margin-bottom: 0.5rem; border-left: 3px solid {risk_color};">
                <span style="font-weight: bold;">{disease_icon} {disease_name}</span> - 
                <span style="color: {risk_color};">{pred.risk_level} Risk ({pred.probability*100:.1f}%)</span> - 
                <span style="color: #666; font-size: 0.9rem;">{pred.created_at.strftime('%Y-%m-%d %H:%M')}</span>
            </div>
            """,
                        unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🔄 Reset Predictions")
    if st.button("Clear All Predictions", use_container_width=True):
        st.session_state.predictions = {}
        st.success("All predictions have been cleared.")
        st.rerun()
