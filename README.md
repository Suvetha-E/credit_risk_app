Credit Risk Analysis with GenAI Explanation & Alerts

This project is an end-to-end Credit Risk Assessment System that combines
✔ Machine Learning (TabTransformer model)
✔ Natural Language Processing
✔ Generative AI (LLM-based explanations)
✔ Streamlit UI

to generate interpretable credit risk scores, human-readable explanations, and automatic alerts based on financial data, news, and filings.

🚀 Features
🔹 1. Interactive Web App (Streamlit)

Input financial & borrower data

Select risk level (LOW / MEDIUM / HIGH)

Provide additional news or filing text

View AI-generated Risk Explanation

View Alerts generated from:

Risk score

News sentiment

Filing red flags

🔹 2. AI-Generated Risk Explanations

Uses an LLM to convert structured data + news into a clear explanation.

🔹 3. Machine Learning Model (TabTransformer)

Trained on credit risk datasets

Model saved as: tabtransformer_credit_model.pth

Custom training script included

🔹 4. Full Data Pipeline

Includes:

Preprocessing

Cleaning

Encoding

Model training

📂 Project Structure
CREDIT_RISK/
│── app.py                           # Streamlit user interface
│── genai_risk_explanation.py        # AI explanation + alerts logic
│── preprocess_pipeline.py            # Data preprocessing code
│── train_transformer.py              # Model training script
│── tabtransformer_credit_model.pth   # Saved credit risk model
│── credit_risk_cleaned.csv           # Clean dataset
│── german_credit_data.csv            # Additional dataset
│── requirements.txt                  # Required Python packages
│── README.md                         # (THIS FILE)
│── .streamlit/
│      └── secrets.toml               # API keys

🛠️ Installation & Running the App
1️⃣ Create a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate

2️⃣ Install dependencies
pip install -r requirements.txt


If Streamlit is missing:

pip install streamlit

3️⃣ Set your OpenAI API key

Inside the folder:

CREDIT_RISK/.streamlit/secrets.toml


Add:

OPENAI_API_KEY = "openai_api_key"

4️⃣ Run the Streamlit app
streamlit run app.py


The app will open at:

http://localhost:8501

📊 How the System Works
1. User enters structured data

Income, DTI, credit history, etc.

2. User enters additional text

Example: news about employer, market updates.

3. Model produces risk classification

(High / Medium / Low)

4. LLM converts model output into a human explanation
5. Alerts are generated automatically

High risk score

Negative news sentiment

Filing red flags

🧠 Technologies Used
Component	Tech
UI	Streamlit
ML Model	TabTransformer (PyTorch)
AI Explanations	OpenAI LLM
Preprocessing	Pandas, Scikit-Learn
Deployment	Streamlit Cloud / Local
📈 Future Improvements

Deploy full backend API

Improve interpretability (SHAP values)

Add credit score prediction

Add loan approval recommendation