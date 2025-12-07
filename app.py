import streamlit as st
from genai_risk_explanation import generate_risk_explanation, check_alerts

st.title("Credit Risk GenAI Explanation & Alerts")

# User Inputs
risk_level = st.selectbox("Select Risk Level", ["LOW", "MEDIUM", "HIGH"], index=2)

structured_data = st.text_area(
    "Structural Data (income, dti, history, etc.)",
    "Income: $50,000, DTI: 0.45, Credit History: 5 years"
)

additional_text = st.text_area(
    "Additional Context (news or filings)",
    "Negative news about borrower's employer."
)

risk_score = st.slider("Risk Score (0–1 scale):", 0.0, 1.0, 0.9)

news_sentiment = st.selectbox("News Sentiment", ["positive", "neutral", "negative"], index=2)

filing_flags = st.multiselect(
    "Filing Flags",
    options=['red flag', 'warning sign', 'none'],
    default=['red flag']
)

# Generate Explanation
if st.button("Generate Risk Explanation"):
    explanation = generate_risk_explanation(
        structured_data,
        risk_level,
        additional_text,
        st.secrets["openai_api_key"]
    )
    st.markdown("### 🧾 Risk Explanation:")
    st.write(explanation)

# Alerts
alerts = check_alerts(risk_score, news_sentiment, filing_flags)
st.markdown("### ⚠️ Alerts:")

if alerts:
    for alert in alerts:
        st.warning(alert)
else:
    st.info("No alerts detected.")
