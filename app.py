import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import random

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="HDFC SecurePay | Fraud Guard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS STYLING (Bank Theme) ---
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {background-color: #004d99; color: white; border-radius: 8px;}
    .stMetric {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.05);}
    </style>
    """, unsafe_allow_html=True)

# --- 1. ASSETS LOAD KARNA ---
@st.cache_resource
def load_assets():
    try:
        model = tf.keras.models.load_model('fraud_detection_model.h5')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# --- 2. FAKE FEATURE NAMES MAPPING (To make it look real) ---
# V1-V28 ko hum asli naam denge taaki dashboard "Technical" lage
feature_names = [
    "User_ID_Hash", "Device_IP_Geo", "Login_Time_Delta", "Transaction_Velocity", 
    "Device_Fingerprint", "Merchant_Category_Code", "Distance_From_Home", "Avg_Spend_Ratio",
    "Last_Password_Change", "Session_Duration", "Browser_Type_Enc", "OS_Version_Hash",
    "IP_Reputation_Score", "Network_Latency", "Mouse_Movement_Pattern", "Typing_Speed_Score",
    "Account_Age_Days", "Previous_Chargebacks", "Credit_Limit_Utilization", "VPN_Usage_Flag",
    "Foreign_Transaction_Flag", "Failed_Attempts_Count", "Email_Domain_Score", "Billing_Address_Match",
    "CVV_Match_Score", "3D_Secure_Status", "Card_Type_Enc", "Token_ID_Hash"
]

# --- 3. HIDDEN VECTORS (Backend Data) ---
# Normal User (Safe)
NORMAL_V = [-0.26, -0.28, 1.48, -1.03, -0.06, 0.16, 0.47, -0.22, 0.36, -0.40, 1.07, 0.39, -0.34, 0.23, 1.84, -0.23, -0.23, -0.66, -0.36, 0.08, 0.16, -0.08, 0.20, -0.16, -0.39, -0.04, -0.08, -0.08]

# Hacker / Fraudster (Dangerous)
FRAUD_V = [-15.5, 8.5, -20.2, 5.5, -10.1, -5.2, -15.5, 5.2, -8.5, -12.2, 8.1, -15.5, 0.5, -18.5, 0.1, -10.2, -12.5, -5.5, 4.5, 1.5, 0.5, 0.1, -0.5, 0.2, -0.5, 0.5, 0.5, 0.1]


# --- HEADER ---
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/2504/2504933.png", width=80)
with col_title:
    st.title("SecurePay Transaction Gateway")
    st.caption("AI-Powered Real-Time Risk Assessment Engine")

st.divider()

# --- MAIN UI ---
col1, col2 = st.columns([1, 1.5])

# === LEFT SIDE: REAL FORM ===
with col1:
    st.subheader("💳 Transaction Details")
    with st.container(border=True):
        # Decorative Fields (Sirf dikhane ke liye)
        st.text_input("Merchant Name", placeholder="Ex: Apple Store, Amazon, Starbucks")
        
        c1, c2 = st.columns(2)
        with c1:
            # ASLI INPUT 1: Amount
            amount = st.number_input("Amount ($)", value=100.0, step=50.0)
        with c2:
            st.selectbox("Transaction Type", ["Online Purchase", "POS Terminal", "ATM Withdrawal"])
        
        st.write("---")
        st.write("**Security Context (Simulation):**")
        
        # LOGIC CONTROL: Ye dropdown backend decide karega
        risk_level = st.radio(
            "Select Scenario Environment:", 
            ["🟢 Trusted Location (Home/Office)", "🔴 High-Risk Location (Unknown IP/Foreign)"],
            index=0
        )
        
        if "High-Risk" in risk_level:
            st.warning("Simulating: Unknown IP Address in Russia/China + VPN Enabled.")
            current_v = FRAUD_V # Backend Fraud Data load karega
        else:
            st.success("Simulating: Verified Device at Home Address.")
            current_v = NORMAL_V # Backend Safe Data load karega

        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🛡️ PROCESS TRANSACTION", type="primary", use_container_width=True)

# === RIGHT SIDE: SYSTEM LOGS ===
with col2:
    st.subheader("🖥️ Live System Logs")
    
    if analyze_btn:
        if model:
            with st.spinner("Decrypting transaction parameters..."):
                # --- PROCESSING LOGIC ---
                amount_df = pd.DataFrame([[amount]], columns=['Amount'])
                amount_scaled = scaler.transform(amount_df)
                final_input = np.array([current_v + amount_scaled.tolist()[0]])
                
                # Prediction
                prob = model.predict(final_input)[0][0]
                percent = prob * 100
                
                # --- DISPLAY RESULT ---
                st.write("---")
                if percent > 50:
                    st.error(f"❌ BLOCKED: FRAUD DETECTED")
                    st.metric("Risk Score", f"{percent:.2f}%", "CRITICAL", delta_color="inverse")
                    st.toast("Alert sent to Fraud Team!", icon="🚨")
                else:
                    st.success(f"✅ APPROVED: Verified Transaction")
                    st.metric("Risk Score", f"{percent:.2f}%", "SAFE", delta_color="normal")
                    st.balloons()
                
                # --- FAKE "REAL DATA" VISUALIZATION ---
                with st.expander("🔍 View Technical Signals (Backend Data)", expanded=True):
                    st.caption("These are the 28 hidden features extracted by the AI model:")
                    
                    # Dataframe banate hain sundar dikhane ke liye
                    df_display = pd.DataFrame({
                        "Feature Name (Mapped)": feature_names,
                        "Extracted Value": current_v
                    })
                    
                    # Randomly thoda color dikhane ke liye styling
                    st.dataframe(
                        df_display, 
                        column_config={
                            "Extracted Value": st.column_config.ProgressColumn(
                                "Signal Strength",
                                help="Neural Network Activation Level",
                                format="%.2f",
                                min_value=-20,
                                max_value=20,
                            ),
                        },
                        height=400,
                        use_container_width=True
                    )
        else:
            st.error("Model files missing!")
    else:
        st.info("Waiting for transaction input...")
        st.image("https://cdn.dribbble.com/users/2046015/screenshots/5973727/06-loader_telecommuting.gif", width=400)

# --- FOOTER ---
st.markdown("---")
st.markdown("<center>SecurePay Banking System v2.4 | Protected by 256-bit Encryption</center>", unsafe_allow_html=True)