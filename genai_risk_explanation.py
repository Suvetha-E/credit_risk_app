from openai import OpenAI
import logging

logging.basicConfig(filename='genai_module.log', level=logging.INFO, format='%(asctime)s - %(message)s')

client = OpenAI()   # new client

prompt_template = """
You are a credit risk analyst. Given the following data:

Structural data:
{structural_data}

Model Risk Level: {risk_level}

Additional context:
{additional_text}

Provide:
- A summary of the risk
- The main reasons for this risk
- 3 suggestions to improve the credit profile

Please be clear and concise.
"""

def generate_risk_explanation(structured_data, risk_level, additional_text, api_key):
    prompt = prompt_template.format(
        structural_data=structured_data,
        risk_level=risk_level,
        additional_text=additional_text
    )

    client.api_key = api_key  # set key

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.2,
        max_tokens=350
    )

    # NEW API — correct way to extract content
    explanation = response.choices[0].message.content

    logging.info(f"Risk explanation generated for risk level: {risk_level}")
    return explanation


def check_alerts(risk_score, news_sentiment, filing_flags, risk_threshold=0.8):
    alerts = []
    if risk_score > risk_threshold:
        alerts.append("⚠️ Risk level is HIGH.")
    if news_sentiment == 'negative':
        alerts.append("⚠️ Negative news sentiment detected.")
    if 'red flag' in filing_flags:
        alerts.append("⚠️ Critical filing flag present.")
    logging.info(f"Alerts generated: {alerts}")
    return alerts
