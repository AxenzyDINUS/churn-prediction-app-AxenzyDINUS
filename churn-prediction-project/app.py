import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import pickle

# Set page config
st.set_page_config(
    page_title="Prediksi Churn Pelanggan",
    page_icon="📊",
    layout="centered"
)

# Title
st.title("📱 Prediksi Churn Pelanggan Telco")
st.write("Aplikasi untuk memprediksi apakah pelanggan akan berhenti berlangganan")

# ==================== LOAD MODEL DENGAN MULTI-PATH ====================
@st.cache_resource
def load_model_and_preprocessor():
    """
    Coba load model dari berbagai lokasi yang mungkin
    """
    # Daftar semua lokasi yang mungkin untuk model
    model_paths = [
        'best_churn_model.pkl',                    # Root folder
        'notebooks/best_churn_model.pkl',          # Notebooks folder
        './best_churn_model.pkl',                  # Current directory
        '../best_churn_model.pkl',                 # Parent directory
        'churn-prediction-project/best_churn_model.pkl',  # Project folder structure
    ]
    
    # Daftar lokasi untuk preprocessor
    preprocessor_paths = [
        'preprocessor.pkl',
        'notebooks/preprocessor.pkl',
        './preprocessor.pkl',
        'feature_names.pkl',
        'notebooks/feature_names.pkl'
    ]
    
    model = None
    preprocessor = None
    model_location = ""
    preprocessor_location = ""
    
    # Cari model
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path) if path.endswith('.pkl') else None
                model_location = path
                st.sidebar.info(f"Model ditemukan di: {path}")
                break
            except Exception as e:
                st.sidebar.warning(f"Gagal load dari {path}: {str(e)[:50]}...")
    
    # Cari preprocessor/feature names
    for path in preprocessor_paths:
        if os.path.exists(path):
            try:
                if path.endswith('feature_names.pkl'):
                    with open(path, 'rb') as f:
                        preprocessor = pickle.load(f)
                else:
                    preprocessor = joblib.load(path)
                preprocessor_location = path
                st.sidebar.info(f"Preprocessor ditemukan di: {path}")
                break
            except Exception as e:
                st.sidebar.warning(f"Gagal load preprocessor dari {path}: {str(e)[:50]}...")
    
    # Jika tidak ditemukan
    if not model:
        st.error("""
        ❌ **Model tidak ditemukan!** 
        
        **Lokasi yang dicari:**
        1. `best_churn_model.pkl` (root folder)
        2. `notebooks/best_churn_model.pkl`
        3. `./best_churn_model.pkl`
        
        **Solusi:**
        1. Pastikan file model ada di salah satu lokasi di atas
        2. Jika di GitHub, pastikan file `.pkl` tidak di-exclude oleh `.gitignore`
        3. Upload file ke folder yang sama dengan `app.py`
        """)
        
        # Debug info
        with st.expander("🔍 Debug Information"):
            st.write("**File yang ada di direktori:**")
            try:
                files = os.listdir('.')
                st.write(files[:20])  # Tampilkan 20 file pertama
            except:
                st.write("Tidak bisa membaca direktori")
                
            st.write("**File di notebooks/:**")
            try:
                notebook_files = os.listdir('notebooks') if os.path.exists('notebooks') else []
                st.write(notebook_files[:20])
            except:
                st.write("Folder notebooks tidak ada atau tidak bisa diakses")
    
    return model, preprocessor, model_location, preprocessor_location

# Load model dan preprocessor
model, preprocessor, model_loc, preproc_loc = load_model_and_preprocessor()

if model:
    st.sidebar.success(f"✅ Model siap digunakan ({model_loc})")
if preprocessor:
    st.sidebar.success(f"✅ Preprocessor siap ({preproc_loc})")

# ==================== INPUT FORM ====================
st.sidebar.header("Input Data")

# Simple form
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (bulan)", 0, 72, 24)
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
total_charges = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, tenure * monthly_charges)

# Default values for other features
multiple_lines = "No"
online_security = "No"
online_backup = "No"
device_protection = "No"
tech_support = "No"
streaming_tv = "No"
streaming_movies = "No"

# Create input dataframe
input_data = {
    'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

input_df = pd.DataFrame([input_data])

# ==================== DISPLAY INPUT ====================
st.subheader("Data Input")
st.write(input_df)

# ==================== PREDICTION ====================
if st.button("Prediksi Churn"):
    if model:
        try:
            # Transform input jika ada preprocessor
            if preprocessor:
                try:
                    # Coba transform dengan preprocessor
                    if hasattr(preprocessor, 'transform'):
                        input_transformed = preprocessor.transform(input_df)
                    elif isinstance(preprocessor, list) or isinstance(preprocessor, dict):
                        # Jika preprocessor adalah feature names
                        st.info("Menggunakan feature names untuk validasi")
                except Exception as e:
                    st.warning(f"Transformasi data gagal: {e}. Menggunakan data asli.")
                    input_transformed = input_df
            else:
                input_transformed = input_df
            
            # Predict
            prediction = model.predict(input_transformed)[0]
            probabilities = model.predict_proba(input_transformed)[0]
            
            # Display results
            st.subheader("Hasil Prediksi")
            
            if prediction == 1:
                st.error("## 🔴 CHURN: YA")
                st.write("Pelanggan berpotensi berhenti berlangganan")
            else:
                st.success("## 🟢 CHURN: TIDAK")
                st.write("Pelanggan cenderung tetap berlangganan")
            
            # Probabilities
            st.subheader("Probabilitas")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Tidak Churn", f"{probabilities[0]*100:.1f}%")
            with col2:
                st.metric("Churn", f"{probabilities[1]*100:.1f}%")
            
            # Simple bar chart
            prob_df = pd.DataFrame({
                'Status': ['Tidak Churn', 'Churn'],
                'Probabilitas': [probabilities[0]*100, probabilities[1]*100]
            })
            
            st.bar_chart(prob_df.set_index('Status'))
            
            # Recommendations
            st.subheader("Rekomendasi")
            if prediction == 1:
                st.warning("""
                **Tindakan yang disarankan:**
                1. Hubungi pelanggan untuk feedback
                2. Tawarkan diskon khusus
                3. Perbaiki layanan yang dikeluhkan
                """)
            else:
                st.info("""
                **Strategi retensi:**
                1. Program loyalitas
                2. Tawaran eksklusif
                3. Komunikasi rutin
                """)
                
        except Exception as e:
            st.error(f"Error saat prediksi: {str(e)}")
            with st.expander("Detail Error"):
                st.write("Input data:", input_df)
                st.write("Model type:", type(model))
                if hasattr(model, 'feature_names_in_'):
                    st.write("Model features:", model.feature_names_in_)
    else:
        st.error("Model belum dimuat. Pastikan file model ada.")

# ==================== DEBUG SECTION ====================
with st.sidebar.expander("🔧 Debug & Info"):
    st.write("**Model Location:**", model_loc if model_loc else "Not found")
    st.write("**Preprocessor Location:**", preproc_loc if preproc_loc else "Not found")
    st.write("**Current Directory:**", os.getcwd())
    
    if st.button("Check Files"):
        st.write("**Files in current dir:**")
        try:
            files = os.listdir('.')
            for f in files:
                if f.endswith('.pkl') or f.endswith('.py'):
                    st.write(f"- {f}")
        except:
            st.write("Cannot list files")

# ==================== FOOTER ====================
st.markdown("---")
st.caption("Proyek UAS Bengkel Koding Data Science - Universitas Dian Nuswantoro")