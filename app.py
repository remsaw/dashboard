import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc
from scipy.stats import chi2_contingency, ttest_ind

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Clinical & Administrative Dashboard", layout="wide")
st.markdown("<h1 style='text-align:center;'>🏥 Clinical & Administrative Analytics Dashboard</h1>", unsafe_allow_html=True)

# ----------------------------
# Load Data (cached)
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("medical_dataset.csv")
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
# TAB 1: Descriptive Analytics (Top 5 Only)
# ----------------------------
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Patients", f"{len(df):,}")
    col2.metric("🎂 Avg Age", f"{df['Age'].mean():.1f}")
    col3.metric("🔄 Readmission Rate", f"{df['Readmission30Days'].mean()*100:.1f}%")
    col4.metric("⭐ Avg Satisfaction", f"{df['SatisfactionScore'].mean():.1f}/5")

    # Top 5 Diagnoses
    diag_counts = df['Diagnosis'].value_counts().head(5).reset_index()
    diag_counts.columns = ["Diagnosis", "Count"]
    fig_diag = px.bar(diag_counts, x="Count", y="Diagnosis", orientation='h',
                      title="Top 5 Diagnoses by Patient Count",
                      color_discrete_sequence=["#0072B2"])
    st.plotly_chart(fig_diag, use_container_width=True)

    # Vital Signs: Top 5 Diagnoses Only
    top5_diag = diag_counts['Diagnosis'].tolist()
    sample_df = df[df['Diagnosis'].isin(top5_diag)].sample(n=min(1000, len(df)), random_state=42)
    colA, colB = st.columns(2)
    with colA:
        fig_bp = px.box(sample_df, x='Diagnosis', y='BloodPressure_Sys', 
                        title="Systolic BP: Top 5 Diagnoses")
        st.plotly_chart(fig_bp, use_container_width=True)
    with colB:
        fig_gluc = px.box(sample_df, x='Diagnosis', y='Glucose', 
                          title="Glucose Level: Top 5 Diagnoses")
        st.plotly_chart(fig_gluc, use_container_width=True)

    # Top 5 Departments by Wait Time
    dept_wait = df.groupby('HospitalDept').agg(
        AvgWait=('WaitTime', 'mean'),
        Count=('PatientID', 'count')
    ).reset_index().sort_values('AvgWait', ascending=False).head(5)
    fig_wait = px.bar(dept_wait, x='AvgWait', y='HospitalDept', orientation='h',
                      title="Top 5 Departments by Average Wait Time (Minutes)",
                      color_discrete_sequence=["#D55E00"])
    st.plotly_chart(fig_wait, use_container_width=True)

# ----------------------------
# TAB 2: Inferential Analytics (Top Insights Only)
# ----------------------------
with tab2:
    st.subheader("🔍 Key Statistical Insights")

    # Top 5 Correlations
    num_cols = ['Age', 'BMI', 'BloodPressure_Sys', 'Cholesterol', 'Glucose',
                'WaitTime', 'VisitDuration', 'TreatmentCost', 'SatisfactionScore']
    corr = df[num_cols].corr()
    corr_unstack = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).unstack().dropna()
    corr_unstack = corr_unstack.abs().sort_values(ascending=False).head(5)
    st.write("**Top 5 Strongest Correlations:**")
    for (var1, var2), val in corr_unstack.items():
        st.write(f"- **{var1} ↔ {var2}**: {val:.2f}")

    # Chi-square: Smoking vs Diagnosis
    st.markdown("### 🚬 Smoking & Diagnosis Association")
    contingency = pd.crosstab(df['Smoker'], df['Diagnosis'])
    chi2, p, _, _ = chi2_contingency(contingency)
    st.info(f"**Chi-square p-value = {p:.4f}** → {'Significant association' if p < 0.05 else 'No significant association'}")

    # T-test: Highest vs Lowest BMI diagnosis
    bmi_by_diag = df.groupby('Diagnosis')['BMI'].mean().sort_values(ascending=False)
    if len(bmi_by_diag) >= 2:
        high_diag = bmi_by_diag.index[0]
        low_diag = bmi_by_diag.index[-1]
        bmi_high = df[df['Diagnosis'] == high_diag]['BMI']
        bmi_low = df[df['Diagnosis'] == low_diag]['BMI']
        if len(bmi_high) > 1 and len(bmi_low) > 1:
            t_stat, p_t = ttest_ind(bmi_high, bmi_low, equal_var=False)
            st.info(f"**T-test: {high_diag} vs {low_diag} → p = {p_t:.4f}**")

    # Top 5 Medication Adherence by Readmission
    adherence_readmit = df.groupby('MedicationAdherence')['Readmission30Days'].mean().sort_values(ascending=False).head(5).reset_index()
    fig_adh = px.bar(adherence_readmit, x='MedicationAdherence', y='Readmission30Days',
                     title="Top 5 Medication Adherence Levels by Readmission Rate",
                     labels={'Readmission30Days': 'Readmission Rate'},
                     color_discrete_sequence=["#009E73"])
    st.plotly_chart(fig_adh, use_container_width=True)

# ----------------------------
# TAB 3: Predictive Analytics (Top 5 Features & Risks)
# ----------------------------
with tab3:
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
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        st.text("📋 Model Performance (Test Set):")
        st.text(classification_report(y_test, y_pred, zero_division=0))

        # ROC Curve
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        roc_data = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        fig_roc = px.area(roc_data, x="FPR", y="TPR", title=f"ROC Curve (AUC={roc_auc:.2f})",
                          labels=dict(FPR='False Positive Rate', TPR='True Positive Rate'))
        st.plotly_chart(fig_roc, use_container_width=True)

        # Top 5 Feature Importance
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)[:5]
        importances = importances.reset_index()
        importances.columns = ["Feature", "Importance"]
        fig_imp = px.bar(importances, x="Importance", y="Feature", orientation='h',
                         title="Top 5 Predictors of 30-Day Readmission")
        st.plotly_chart(fig_imp, use_container_width=True)

        # Top 5 High-Risk Patients
        df['ReadmissionRisk'] = model.predict_proba(X)[:, 1]
        st.subheader("⚠️ Top 5 Highest-Risk Patients")
        st.dataframe(df[['PatientID', 'Diagnosis', 'ReadmissionRisk']].sort_values('ReadmissionRisk', ascending=False).head(5))

# ----------------------------
# TAB 4: Data (Top 5 Rows)
# ----------------------------
with tab4:
    st.subheader("📋 Sample Patient Data (Top 5 Rows)")
    st.dataframe(df.head(5))

    # Full data download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Full Filtered Data as CSV", csv, "filtered_data.csv", "text/csv")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("💡 Dashboard built with Streamlit | Showing top 5 insights for clarity and actionability")
