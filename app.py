import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from scipy.stats import chi2_contingency, ttest_ind
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------
# Page Config & Theming
# ----------------------------
st.set_page_config(page_title="Clinical & Administrative Dashboard", layout="wide")
st.markdown("<h1 style='text-align:center;'>🏥 Clinical & Administrative Analytics Dashboard</h1>", unsafe_allow_html=True)
st.caption("💡 Explore patient data with descriptive, inferential, and predictive insights.")

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("medical_dataset.csv")
    # Convert binary/categorical
    df['Readmission30Days'] = df['Readmission30Days'].map({'Yes': 1, 'No': 0})
    df['Smoker'] = df['Smoker'].map({'Yes': 1, 'No': 0})
    if "AdmissionDate" in df.columns:
        df['AdmissionDate'] = pd.to_datetime(df['AdmissionDate'], errors="coerce")
    return df

df = load_data()

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("🔍 Filters")
diagnosis_filter = st.sidebar.multiselect("Diagnosis", options=df['Diagnosis'].unique(), default=df['Diagnosis'].unique())
dept_filter = st.sidebar.multiselect("Department", options=df['HospitalDept'].unique(), default=df['HospitalDept'].unique())
age_group = st.sidebar.selectbox("Age Group", ["All", "18-30", "31-50", "51-70", "71+"])

# Apply filters
if age_group == "18-30":
    df = df[(df['Age'] >= 18) & (df['Age'] <= 30)]
elif age_group == "31-50":
    df = df[(df['Age'] > 30) & (df['Age'] <= 50)]
elif age_group == "51-70":
    df = df[(df['Age'] > 50) & (df['Age'] <= 70)]
elif age_group == "71+":
    df = df[df['Age'] > 70]

df = df[df['Diagnosis'].isin(diagnosis_filter)]
df = df[df['HospitalDept'].isin(dept_filter)]

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Descriptive", "🔍 Inferential", "🤖 Predictive", "📋 Data"])

# ----------------------------
# TAB 1: Descriptive Analytics
# ----------------------------
with tab1:
    st.subheader("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Patients", len(df))
    col2.metric("🎂 Avg Age", f"{df['Age'].mean():.1f}")
    col3.metric("🔄 Readmission Rate", f"{df['Readmission30Days'].mean()*100:.1f}%")
    col4.metric("⭐ Avg Satisfaction", f"{df['SatisfactionScore'].mean():.1f}/5")

    # Admissions Over Time
    if "AdmissionDate" in df.columns:
        st.subheader("📅 Admissions Over Time")
        admissions = df.groupby(df['AdmissionDate'].dt.to_period("M")).size()
        fig_time = px.line(admissions, title="Monthly Admissions")
        st.plotly_chart(fig_time, use_container_width=True)

    # Demographics
    st.subheader("👤 Demographics")
    cols = st.columns(2)
    with cols[0]:
        fig_age = px.histogram(df, x="Age", nbins=20, title="Age Distribution")
        st.plotly_chart(fig_age, use_container_width=True)
    with cols[1]:
        if "Gender" in df.columns:
            fig_gender = px.pie(df, names="Gender", title="Gender Distribution")
            st.plotly_chart(fig_gender, use_container_width=True)

    # Top Diagnoses by Cost
    st.subheader("💰 Top 5 Diagnoses by Avg Treatment Cost")
    top_diag_cost = df.groupby("Diagnosis")["TreatmentCost"].mean().sort_values(ascending=False).head(5)
    fig_top_cost = px.bar(top_diag_cost, title="Top 5 Diagnoses by Cost")
    st.plotly_chart(fig_top_cost, use_container_width=True)

    # Vital Signs
    st.subheader("📈 Vital Signs by Diagnosis")
    cols = st.columns(2)
    with cols[0]:
        fig_bp = px.box(df, x='Diagnosis', y='BloodPressure_Sys', title="Systolic BP by Diagnosis")
        st.plotly_chart(fig_bp, use_container_width=True)
    with cols[1]:
        fig_gluc = px.box(df, x='Diagnosis', y='Glucose', title="Glucose Level by Diagnosis")
        st.plotly_chart(fig_gluc, use_container_width=True)

    # Operational Metrics
    st.subheader("🏥 Operational Metrics")
    fig_wait = px.scatter(df, x='WaitTime', y='SatisfactionScore', color='Diagnosis',
                          title="Wait Time vs Satisfaction")
    st.plotly_chart(fig_wait, use_container_width=True)

# ----------------------------
# TAB 2: Inferential Analytics
# ----------------------------
with tab2:
    st.subheader("🔗 Correlation Heatmap (Numeric Features)")
    num_cols = ['Age', 'BMI', 'BloodPressure_Sys', 'Cholesterol', 'Glucose',
                'WaitTime', 'VisitDuration', 'TreatmentCost', 'SatisfactionScore']
    corr = df[num_cols].corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("📊 Statistical Tests")
    st.write("🔹 **Chi-square Test**: Is smoking associated with diagnosis?")
    contingency = pd.crosstab(df['Smoker'], df['Diagnosis'])
    chi2, p, _, _ = chi2_contingency(contingency)
    st.info(f"p-value = {p:.4f} → {'Significant' if p < 0.05 else 'Not significant'}")

    if 'Hypertension' in df['Diagnosis'].values and 'Healthy' in df['Diagnosis'].values:
        bmi_hyp = df[df['Diagnosis'] == 'Hypertension']['BMI']
        bmi_healthy = df[df['Diagnosis'] == 'Healthy']['BMI']
        t_stat, p_t = ttest_ind(bmi_hyp, bmi_healthy, equal_var=False)
        st.info(f"🔹 **T-test**: BMI (Hypertension vs Healthy) → p = {p_t:.4f}")

    # Medication Adherence vs Readmission
    st.subheader("💊 Medication Adherence Impact")
    adherence_readmit = df.groupby('MedicationAdherence')['Readmission30Days'].mean().reset_index()
    fig_adh = px.bar(adherence_readmit, x='MedicationAdherence', y='Readmission30Days',
                     title="Readmission Rate by Medication Adherence",
                     labels={'Readmission30Days': 'Readmission Rate'})
    st.plotly_chart(fig_adh, use_container_width=True)

# ----------------------------
# TAB 3: Predictive Analytics
# ----------------------------
with tab3:
    st.subheader("🔮 30-Day Readmission Risk Prediction")
    df_model = df.copy()
    df_model = pd.get_dummies(df_model, columns=['Diagnosis', 'MedicationAdherence', 'Gender'], drop_first=True)

    feature_cols = ['Age', 'BMI', 'BloodPressure_Sys', 'Cholesterol', 'Glucose',
                    'LengthOfStay', 'WaitTime', 'TreatmentCost']
    for col in df_model.columns:
        if col.startswith(('Diagnosis_', 'MedicationAdherence_', 'Gender_')):
            feature_cols.append(col)

    X = df_model[feature_cols].fillna(0)
    y = df_model['Readmission30Days']

    if len(y.unique()) < 2:
        st.warning("⚠️ Not enough readmission cases to train model.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Performance
        y_pred = model.predict(X_test)
        st.text("📋 Model Performance (Test Set):")
        st.text(classification_report(y_test, y_pred, zero_division=0))

        # ROC Curve
        y_prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig_roc = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC={roc_auc:.2f})",
                          labels=dict(x='False Positive Rate', y='True Positive Rate'))
        st.plotly_chart(fig_roc, use_container_width=True)

        # Feature Importance
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
        fig_imp = px.bar(importances, orientation='h', title="Top 10 Predictors of Readmission")
        st.plotly_chart(fig_imp, use_container_width=True)

        # High-risk patients
        df['ReadmissionRisk'] = model.predict_proba(X)[:, 1]
        df['RiskGroup'] = pd.cut(df['ReadmissionRisk'], bins=[0,0.33,0.66,1], labels=["Low","Medium","High"])
        st.subheader("⚠️ High-Risk Patients (Top 10)")
        st.dataframe(df[['PatientID', 'Diagnosis', 'ReadmissionRisk','RiskGroup']].sort_values('ReadmissionRisk', ascending=False).head(10))

# ----------------------------
# TAB 4: Raw Data
# ----------------------------
with tab4:
    st.subheader("📋 Raw Patient Data (Filtered)")
    st.dataframe(df)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered Data as CSV", csv, "filtered_data.csv", "text/csv")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("💡 Dashboard built with Streamlit | Data: medical_dataset.csv")
