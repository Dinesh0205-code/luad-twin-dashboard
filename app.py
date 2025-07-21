
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(page_title="Digital Twin Simulator", layout="wide")
st.title("🧬 Digital Twin Clinical Predictor")

patient_df = pd.read_csv('Patient_vector.csv')
model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')

vector_cols = [col for col in patient_df.columns if col.startswith('PC')]
age_cols = [col for col in patient_df.columns if col.startswith('age_')]
stage_cols = [col for col in patient_df.columns if col.startswith('stage_')]
feature_cols = vector_cols + age_cols + stage_cols

st.sidebar.header("🧪 Input Patient Vector")
input_data = []
for col in feature_cols:
    min_val = float(patient_df[col].min())
    max_val = float(patient_df[col].max())
    mean_val = float(patient_df[col].mean())
    if min_val == max_val:
        st.sidebar.text(f"{col} (constant): {min_val}")
        input_data.append(min_val)
    else:
        val = st.sidebar.slider(col, min_val, max_val, mean_val)
        input_data.append(val)

input_array = np.array(input_data).reshape(1, -1)
input_scaled = scaler.transform(input_array)

prediction = model.predict(input_scaled)
proba = model.predict_proba(input_scaled)[0]
status = "🟢 Alive" if prediction[0] == 0 else "🔴 Dead"

st.subheader("🩺 Prediction Outcome")
st.metric(label="Predicted Vital Status", value=status)
st.write(f"**Probability Alive:** `{proba[0]:.2f}`")
st.write(f"**Probability Dead:** `{proba[1]:.2f}`")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_scaled)
fig, ax = plt.subplots()
shap.summary_plot(shap_values, pd.DataFrame(input_scaled, columns=feature_cols), plot_type='bar', show=False)
st.pyplot(fig)

with st.expander("🧠 SHAP Force Plot"):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig_force = shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        pd.DataFrame(input_scaled, columns=feature_cols).iloc[0],
        matplotlib=True
    )
    st.pyplot(fig_force)

st.subheader("📋 Input Vector")
st.dataframe(pd.DataFrame(input_array, columns=feature_cols))
