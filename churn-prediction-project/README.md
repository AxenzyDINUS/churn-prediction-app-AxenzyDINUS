# 🎓 Customer Churn Prediction - UAS Bengkel Koding

**Nama:** Singgih Arya Mahendra  
**NIM:** A11.2022.14246  
**Universitas:** Universitas Dian Nuswantoro  
**Mata Kuliah:** Bengkel Koding Data Science

## 📋 Overview
Proyek prediksi customer churn menggunakan dataset Telco dengan 9 model machine learning dan deployment Streamlit.

## 📊 Hasil Utama
- **Model Terbaik:** Random Forest dengan Hyperparameter Tuning
- **Accuracy:** 0.8456
- **F1-Score:** 0.6432
- **Probabilitas Contoh:** 78.1% (Tidak Churn) vs 21.9% (Churn)

## 🏗️ Struktur Proyek
1. **EDA** - Exploratory Data Analysis
2. **Direct Modeling** - 3 model tanpa preprocessing
3. **Model dengan Preprocessing** - 3 model dengan preprocessing
4. **Hyperparameter Tuning** - Optimasi model terbaik
5. **Deployment** - Aplikasi Streamlit

## 🚀 Cara Menjalankan
```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py