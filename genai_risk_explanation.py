from openai import OpenAI

def generate_risk_explanation(structured_data, risk_level, additional_text, api_key):
    client = OpenAI(api_key=api_key)

    prompt = f"""
    Given the following credit inputs:
    Structured Data: {structured_data}
    Risk Level: {risk_level}
    Additional Context: {additional_text}

    Generate a detailed risk explanation.
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message["content"]


def check_alerts(risk_score, sentiment, flags):
    alerts = []

    if risk_score > 0.7:
        alerts.append("High risk score detected.")
    
    if sentiment == "negative":
        alerts.append("Negative news sentiment flagged.")
    
    if "red flag" in flags:
        alerts.append("Filing red flag detected.")

    return alerts
