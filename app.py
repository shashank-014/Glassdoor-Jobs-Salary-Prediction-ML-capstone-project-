import matplotlib.pyplot as plt
import streamlit as st

from glassdoor_jobs_project_code_export import dataset_overview, load_data, run_modeling

st.set_page_config(page_title="Glassdoor Jobs Salary Prediction", layout="wide")
st.title("Glassdoor Jobs Salary Prediction")
st.caption("I used this app to walk through how I cleaned the job data and estimated salary bands.")

raw = load_data()
results = run_modeling()
overview = dataset_overview(raw)
frame = results["model_frame"]

tab1, tab2, tab3 = st.tabs(["Overview", "EDA", "Modeling"])

with tab1:
    st.subheader("Dataset Snapshot")
    st.dataframe(overview, use_container_width=True)
    st.dataframe(raw.head(10), use_container_width=True)

with tab2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    frame["avg_salary_k"].dropna().plot(kind="hist", bins=25, ax=axes[0], title="Salary distribution (K USD)")
    results["sector_salary"].sort_values().plot(kind="barh", ax=axes[1], title="Top sectors by median salary")
    st.pyplot(fig)
    st.dataframe(
        frame.groupby("Job Title")["avg_salary_k"].median().sort_values(ascending=False).head(10).reset_index(),
        use_container_width=True,
    )

with tab3:
    st.subheader("Model Comparison")
    st.dataframe(results["metrics"], use_container_width=True)
    st.write(f"Best model: **{results['best_model']}**")
    st.subheader("Feature Importance")
    st.dataframe(results["feature_importance"], use_container_width=True)