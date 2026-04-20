import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

st.title("📊 Attendance & Behavior Prediction App")

uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📄 Data Preview")
    st.write(df.head())

    # Check required columns
    required_cols = ['days_present','days_absent','late_count','avg_login_hour','behavior_score','target']

    if all(col in df.columns for col in required_cols):

        X = df[['days_present','days_absent','late_count','avg_login_hour','behavior_score']]
        y = df['target']

        # Train model
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X, y)

        # Predict
        df['prediction'] = model.predict(X)

        # Convert to label
        df['prediction_label'] = df['prediction'].apply(lambda x: "High Risk" if x == 1 else "Low Risk")

        st.subheader("✅ Predictions")
        st.write(df)

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Results",
            csv,
            "predictions.csv",
            "text/csv"
        )

    else:
        st.error(f"❌ File must contain columns: {required_cols}")
