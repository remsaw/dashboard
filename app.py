import logging

# Reduce Streamlit logger noise when running the script directly with `python app.py`.
# Recommended way to run the app is `streamlit run app.py` (see notes below).
logging.getLogger("streamlit").setLevel(logging.ERROR)

import streamlit as st
import warnings

# Suppress the repeated 'missing ScriptRunContext' warning when running
# the app directly with `python app.py`. Prefer `streamlit run app.py`.
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
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

    # Age Distribution & Cost per Diagnosis side-by-side
    # create age bins for coloring (matches sidebar groups)
    bins = [0, 18, 30, 50, 70, 200]
    labels = ['<18', '18-30', '31-50', '51-70', '71+']
    df['age_bin'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)
    # remove categories that have no observations so the legend only shows present bins
    if hasattr(df['age_bin'], 'cat'):
        df['age_bin'] = df['age_bin'].cat.remove_unused_categories()

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### 🎂 Age Distribution")
        # Histogram colored by age bins, overlayed with some transparency
        fig_age = px.histogram(
            df,
            x='Age',
            nbins=20,
            color='age_bin',
            barmode='overlay',
            title="Age Distribution of Patients",
            labels={'age_bin': 'Age Bin'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_age.update_traces(opacity=0.75)

        # Compute counts and percentages per age bin and attach to hovertemplate
        counts_by_bin = df['age_bin'].value_counts().to_dict()
        total_n = len(df)
        perc_by_bin = {k: (v / total_n * 100) for k, v in counts_by_bin.items()}
        for trace in fig_age.data:
            # trace.name contains the age_bin label for that trace
            total = counts_by_bin.get(trace.name, 0)
            percent = perc_by_bin.get(trace.name, 0.0)
            # show the age value (x), count for that bar (y), and total & percent for that age bin
            trace.hovertemplate = (
                f"Age: %{{x}}<br>Count: %{{y}}<br>Total in {trace.name}: {total} ({percent:.1f}%)<extra></extra>"
            )

        # Interactive selection: prefer plotly click events when streamlit_plotly_events is installed
        selected_bins = []
        # Detect optional streamlit_plotly_events at runtime and provide a safe fallback
        try:
            import importlib
            if importlib.util.find_spec("streamlit_plotly_events") is not None:
                from streamlit_plotly_events import plotly_events  # type: ignore
            else:
                plotly_events = None
        except Exception:
            plotly_events = None

        if plotly_events is not None:
            try:
                # plotly_events renders the figure and returns click events
                events = plotly_events(fig_age, click_event=True, key='age_plot')
                if events:
                    for ev in events:
                        curve = ev.get('curveNumber')
                        if curve is not None and curve < len(fig_age.data):
                            name = fig_age.data[curve].name
                            if name not in selected_bins:
                                selected_bins.append(name)
                # maintain selection in session state
                if 'selected_age_bins' not in st.session_state:
                    st.session_state['selected_age_bins'] = []
                if selected_bins:
                    st.session_state['selected_age_bins'] = selected_bins
                if st.button("Clear age selection", key='clear_age_sel'):
                    st.session_state['selected_age_bins'] = []
            except Exception:
                # If something goes wrong with the optional integration, fall back to static rendering
                plotly_events = None

        if plotly_events is None:
            # fallback: show the static figure and provide a multiselect to filter
            st.plotly_chart(fig_age, use_container_width=True)
            options = list(df['age_bin'].cat.categories)
            sel = st.multiselect("Select age bins to filter other charts", options=options, default=st.session_state.get('selected_age_bins', []))
            st.session_state['selected_age_bins'] = sel

        # Small summary box with mean and median and per-bin counts and percentages
        mean_age = df['Age'].mean()
        median_age = df['Age'].median()
        summary_lines = [f"**Mean Age:** {mean_age:.1f}", f"**Median Age:** {median_age:.1f}", "\n**Counts by Age Bin:**"]
        for lab in labels:
            if lab in counts_by_bin:
                pct = perc_by_bin.get(lab, 0.0)
                summary_lines.append(f"- {lab}: {counts_by_bin[lab]} ({pct:.1f}%)")
        st.info('\n'.join(summary_lines))

    # apply selected age bins to filter the right-hand charts and downstream visuals
    selected = st.session_state.get('selected_age_bins', []) if 'selected_age_bins' in st.session_state else []
    if selected:
        df_sel = df[df['age_bin'].isin(selected)]
    else:
        df_sel = df

    with col_right:
        st.markdown("### 💰 Average Treatment Cost per Diagnosis")
        cost_per_diag = df_sel.groupby('Diagnosis')['TreatmentCost'].mean().sort_values(ascending=False).reset_index()
        # Horizontal bar with annotated average cost values
        fig_cost = px.bar(
            cost_per_diag,
            x='TreatmentCost',
            y='Diagnosis',
            orientation='h',
            title="Average Treatment Cost per Diagnosis",
            labels={'TreatmentCost': 'Avg Cost (USD)'},
            color_discrete_sequence=["#F0E442"],
            text='TreatmentCost'
        )
        # Format text labels and move them outside bars for readability
        fig_cost.update_traces(texttemplate='$%{text:.0f}', textposition='outside')
        fig_cost.update_layout(margin=dict(l=0, r=150))
        st.plotly_chart(fig_cost, use_container_width=True)

    # Top 5 Diagnoses (respecting selected age bins)
    diag_counts = df_sel['Diagnosis'].value_counts().head(5).reset_index()
    diag_counts.columns = ["Diagnosis", "Count"]
    fig_diag = px.bar(diag_counts, x="Count", y="Diagnosis", orientation='h',
                      title="Top 5 Diagnoses by Patient Count",
                      color_discrete_sequence=["#0072B2"])
    st.plotly_chart(fig_diag, use_container_width=True)

    # Vital Signs: Top 5 Diagnoses Only
    top5_diag = diag_counts['Diagnosis'].tolist()
    sample_df = df_sel[df_sel['Diagnosis'].isin(top5_diag)].sample(n=min(1000, len(df_sel)), random_state=42)
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
