import streamlit as st
import numpy as np
import requests

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("Credit Card Fraud Detection System")

st.markdown("""
This app uses a trained **Random Forest model (SMOTE balanced)** 
to predict whether a credit card transaction is **fraudulent or legitimate**.
""")

st.info("""
**Note:**  
Due to confidentiality, the original transaction features are not available.  
Instead, anonymized **Principal Component Analysis (PCA)** features (`V1–V28`) are used to represent transaction characteristics.
""")


feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Hour"]

st.subheader("Input Transaction Data")

if st.button("Generate Random Transaction"):
    random_inputs = np.random.normal(0, 1, len(feature_names))
    random_inputs[0] = np.random.uniform(0, 172800)      
    random_inputs[-2] = np.abs(np.random.normal(100, 200)) 
    random_inputs[-1] = (random_inputs[0] / 3600) % 24     
    st.session_state["inputs"] = random_inputs.tolist()
    st.success("Random transaction generated!")
else:
    if "inputs" not in st.session_state:
        st.session_state["inputs"] = [0.0] * len(feature_names)


cols = st.columns(3)
inputs = []

for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        val = st.number_input(
            f"{feature}",
            value=float(st.session_state["inputs"][i]),
            step=0.01,
            format="%.4f"
        )
        inputs.append(val)

st.session_state["inputs"] = inputs


if st.button("Predict Fraud"):
    with st.spinner("Analyzing transaction..."):
        try:
            response = requests.post("http://127.0.0.1:5000/predict", json={"features": inputs})
            result = response.json()

            if "fraud_prediction" in result:
                fraud = result["fraud_prediction"]
                prob = result["fraud_probability"]

                if fraud == 1:
                    st.error(f"Fraudulent Transaction Detected!\n**Fraud Probability:** {prob:.3f}")
                else:
                    st.success(f"Legitimate Transaction\n**Fraud Probability:** {prob:.3f}")
            else:
                st.warning(f"API Error: {result.get('error', 'Unknown issue')}")

        except Exception as e:
            st.error(f"Error connecting to Flask API: {e}")
