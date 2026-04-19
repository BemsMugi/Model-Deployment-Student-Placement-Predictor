import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    with open("best_classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("best_regressor.pkl", "rb") as f:
        reg = pickle.load(f)
    return clf, reg

clf_model, reg_model = load_models()

def engineer_features(df):
    df = df.copy()
    df["academic_score"] = (
        df["ssc_percentage"] + df["hsc_percentage"] +
        df["degree_percentage"] + df["cgpa"] * 20
    ) / 4
    df["experience_score"] = (
        df["internship_count"] + df["live_projects"] +
        df["work_experience_months"] / 6 + df["certifications"]
    )
    return df

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=80)
    st.title("🎓 Student Predictor")
    st.markdown("---")
    page = st.radio(
        label="Pilih Halaman",
        options=["🏠 Home", "🔮 Prediksi", "📊 Data Visualisasi", "ℹ️ Tentang Model"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Model: Decision Tree (Klasifikasi) | Linear Regression (Regresi)")

if page == "🏠 Home":
    st.title("🎓 Student Placement & Salary Predictor")
    st.markdown("### Prediksi peluang penempatan kerja dan estimasi gaji mahasiswa")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Data Training", "5.000 mahasiswa")
    with col2:
        st.metric("Model Klasifikasi", "Decision Tree")
    with col3:
        st.metric("Model Regresi", "Linear Regression")
    st.markdown("---")
    st.markdown("""
    ### Cara Penggunaan
    1. Pergi ke halaman **🔮 Prediksi** di sidebar
    2. Isi data mahasiswa pada form yang tersedia
    3. Klik tombol **Prediksi** untuk melihat hasil
    4. Hasil menampilkan status penempatan + estimasi gaji (jika ditempatkan)
    """)

elif page == "🔮 Prediksi":
    st.title("🔮 Prediksi Placement & Salary")
    st.markdown("Isi data mahasiswa di bawah ini, lalu klik **Prediksi**.")
    st.markdown("---")

    with st.form("prediction_form"):
        st.subheader("📋 Data Akademik")
        col1, col2, col3 = st.columns(3)
        with col1:
            ssc    = st.slider("SSC Percentage (%)", 50, 95, 70)
            hsc    = st.slider("HSC Percentage (%)", 50, 95, 70)
        with col2:
            degree = st.slider("Degree Percentage (%)", 50, 95, 70)
            cgpa   = st.slider("CGPA (skala 10)", 5.0, 10.0, 7.5, step=0.1)
        with col3:
            entrance = st.slider("Entrance Exam Score", 0, 100, 60)
            backlogs = st.number_input("Jumlah Backlogs", 0, 20, 0)

        st.markdown("---")
        st.subheader("💻 Skills & Experience")
        col4, col5, col6 = st.columns(3)
        with col4:
            tech_skill = st.slider("Technical Skill Score", 0, 100, 60)
            soft_skill = st.slider("Soft Skill Score", 0, 100, 60)
        with col5:
            internship = st.number_input("Jumlah Internship", 0, 10, 1)
            live_proj  = st.number_input("Live Projects", 0, 10, 1)
        with col6:
            work_exp   = st.number_input("Work Experience (bulan)", 0, 60, 0)
            certif     = st.number_input("Jumlah Sertifikasi", 0, 20, 2)

        st.markdown("---")
        st.subheader("👤 Informasi Lain")
        col7, col8, col9 = st.columns(3)
        with col7:
            gender     = st.selectbox("Gender", ["Male", "Female"])
        with col8:
            attendance = st.slider("Attendance (%)", 50, 100, 80)
        with col9:
            extra      = st.selectbox("Extracurricular Activities", ["Yes", "No"])

        submitted = st.form_submit_button("🚀 Prediksi Sekarang", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame([{
            "gender":                    1 if gender == "Male" else 0,
            "ssc_percentage":            ssc,
            "hsc_percentage":            hsc,
            "degree_percentage":         degree,
            "cgpa":                      cgpa,
            "entrance_exam_score":       entrance,
            "technical_skill_score":     tech_skill,
            "soft_skill_score":          soft_skill,
            "internship_count":          internship,
            "live_projects":             live_proj,
            "work_experience_months":    work_exp,
            "certifications":            certif,
            "attendance_percentage":     attendance,
            "backlogs":                  backlogs,
            "extracurricular_activities": 1 if extra == "Yes" else 0,
        }])
        input_data = engineer_features(input_data)

        placement_pred = clf_model.predict(input_data)[0]
        placement_prob = clf_model.predict_proba(input_data)[0][1] if hasattr(clf_model, "predict_proba") else None

        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            if placement_pred == 1:
                st.success("✅ **DITEMPATKAN**")
                st.markdown("Mahasiswa ini diprediksi **berhasil mendapatkan penempatan kerja**.")
            else:
                st.error("❌ **TIDAK DITEMPATKAN**")
                st.markdown("Mahasiswa ini diprediksi **belum berhasil mendapatkan penempatan kerja**.")
            if placement_prob is not None:
                st.metric("Probabilitas Ditempatkan", f"{placement_prob*100:.1f}%")

        with col_r2:
            if placement_pred == 1:
                salary_pred = max(reg_model.predict(input_data)[0], 0)
                st.info(f"💰 **Estimasi Gaji: {salary_pred:.2f} LPA**")
                st.metric("Salary Package", f"₹ {salary_pred:.2f} LPA")
            else:
                st.warning("💰 Estimasi gaji tidak tersedia karena mahasiswa diprediksi tidak ditempatkan.")

        st.markdown("---")
        st.subheader("📈 Profil Mahasiswa")
        categories = ["Academic", "Technical", "Soft Skill", "Entrance Exam", "Experience"]
        values = [
            (ssc + hsc + degree + cgpa * 10) / 4,
            tech_skill, soft_skill, entrance,
            min((internship * 10 + live_proj * 5 + work_exp + certif * 5), 100)
        ]
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.barh(categories, [v/100 for v in values],
                       color=["#3498db","#2ecc71","#e74c3c","#9b59b6","#f39c12"])
        ax.set_xlim(0, 1)
        ax.set_title("Profil Skill Mahasiswa")
        for bar, val in zip(bars, values):
            ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2, f"{val:.1f}", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

elif page == "📊 Data Visualisasi":
    st.title("📊 Data Visualisasi Dataset")
    try:
        df = pd.read_csv("B.csv")
        st.success(f"Dataset loaded: {df.shape[0]} baris, {df.shape[1]} kolom")
        tab1, tab2, tab3 = st.tabs(["Distribusi Target", "Korelasi Fitur", "Statistik Deskriptif"])
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Distribusi Placement Status")
                fig, ax = plt.subplots()
                counts = df["placement_status"].value_counts()
                ax.pie(counts, labels=["Tidak Ditempatkan", "Ditempatkan"],
                       autopct="%1.1f%%", colors=["#e74c3c", "#2ecc71"])
                st.pyplot(fig)
            with col2:
                st.subheader("Distribusi Salary (Placed Only)")
                fig, ax = plt.subplots()
                salary = df[df["placement_status"]==1]["salary_package_lpa"]
                ax.hist(salary, bins=30, color="#3498db", edgecolor="white")
                ax.axvline(salary.mean(), color="red", linestyle="--", label=f"Mean: {salary.mean():.2f}")
                ax.legend()
                st.pyplot(fig)
        with tab2:
            fig, ax = plt.subplots(figsize=(12, 8))
            numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["student_id"], errors="ignore")
            sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
        with tab3:
            st.dataframe(df.describe().round(2), use_container_width=True)
    except FileNotFoundError:
        st.error("File B.csv tidak ditemukan.")

elif page == "ℹ️ Tentang Model":
    st.title("ℹ️ Tentang Model")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏆 Model Klasifikasi")
        st.markdown("""
        **Algoritma:** Decision Tree Classifier
        **Task:** Prediksi placement_status (0/1)
        **Accuracy:** 100%
        """)
    with col2:
        st.subheader("📈 Model Regresi")
        st.markdown("""
        **Algoritma:** Linear Regression
        **Task:** Prediksi salary_package_lpa
        **R² Score:** 0.0248
        """)