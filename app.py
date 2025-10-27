# app.py 

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.decomposition import PCA
import pickle
from sentence_transformers import SentenceTransformer
import copy

# --- Page Setup ---
st.set_page_config(
    page_title="Spectra AI: Prompt Anomaly Detector",
    page_icon="🛡️",
    layout="wide"
)

# --- Model Loading Functions ---

@st.cache_resource
def load_embedding_model():
    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

@st.cache_resource
def load_detector_and_data():
    # Load detector, PCA objects, and data
    try:
        with open('trained_detector.pkl', 'rb') as f:
            detector = pickle.load(f)

        with open('pca_model.pkl', 'rb') as f:
            pca = pickle.load(f)

        with open('data.pkl', 'rb') as f:
            data_dict = pickle.load(f)

        return detector, pca, data_dict['normal_2d'], data_dict['anomalous_2d']

    except FileNotFoundError:
        return None, None, None, None
    except KeyError:
        st.error("Error loading data.pkl. Ensure it contains 'normal_2d' and 'anomalous_2d'.")
        return None, None, None, None

@st.cache_data
def load_drift_history():
    # Load the drift history (a list of floats) for the security plot
    try:
        with open('drift_history.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# --- Distance Calculation (for Chi-Square Plot) ---

@st.cache_data
def calculate_normal_distances(_detector):
    # This requires 'normal_data' to be saved in data.pkl
    try:
        with open('data.pkl', 'rb') as f:
            data_dict_full = pickle.load(f)
            if 'normal_data' not in data_dict_full:
                 st.warning("normal_data not found in data.pkl. Chi-Square plot cannot be generated accurately.")
                 return None
            normal_data_orig = data_dict_full['normal_data']
            dists = [_detector._compute_mahalanobis_sq(prompt) for prompt in normal_data_orig]
            return dists
    except Exception:
        return None

# --- Main App Execution ---

st.title("🛡️ Spectra AI: Anomaly Prompt Detector")
st.markdown("This prototype demonstrates **real-time anomaly detection** on LLM prompts using **Linear Algebra** (Mahalanobis Distance) and **Probability** (Chi-Square Test).")

# --- Load Models and Data ---
embedding_model = load_embedding_model()
detector, pca, normal_data_2d, anomalous_data_2d = load_detector_and_data()
drift_history = load_drift_history() 

# --- Global Confing ---
P_VALUE_THRESHOLD = 7.8329e-02 

if detector is None or pca is None or normal_data_2d is None:
    st.error("Model or data files not found/loaded correctly! Please run the full `main_notebook.ipynb` first.")
    st.stop()
else:
    st.success("Anomaly Detector (384D) is loaded and ready.")
    N_DIMENSIONS = detector.dimensions

# Calculate distances for Chi-Square plot
dists_normal_sq = calculate_normal_distances(detector)


# ====================================================================
# 1. INTERACTIVE PROMPT CHECKER
# ====================================================================
st.header("1. 🕵️ Real-Time Prompt Analysis")
st.markdown(f"The system checks the prompt's {N_DIMENSIONS}-dimensional embedding against the 'normal' distribution learned from diverse examples.")

prompt_text = st.text_area("Enter your prompt:", "What is the weather today?")

if st.button("Analyze Prompt"):
    if prompt_text:
        with st.spinner(f"Generating {N_DIMENSIONS}D embedding and analyzing..."):
            new_embedding = embedding_model.encode([prompt_text])[0]
            flag, p_val, dist_sq = detector.predict(new_embedding, P_VALUE_THRESHOLD)

        if flag == "ANOMALOUS":
            st.error(f"**Status: {flag}** - Potential Anomaly Detected!")
            st.markdown(f"""
            - **Squared Mahalanobis Distance:** `{dist_sq:.2f}`
            - **P-Value:** `{p_val:.2e}`

            **Conclusion:** This prompt is a significant statistical outlier (less than `{P_VALUE_THRESHOLD*100:.2f}%` chance of being normal).
            """)
        else:
            st.success(f"**Status: {flag}** - Cleared")
            st.markdown(f"""
            - **Squared Mahalanobis Distance:** `{dist_sq:.2f}`
            - **P-Value:** `{p_val:.4f}`

            **Conclusion:** This prompt fits within the established normal range.
            """)
    else:
        st.error("Please enter a prompt.")

st.divider()

# ====================================================================
# 2. KEY FINDINGS & VISUALIZATIONS
# ====================================================================
st.header("2. 📊 Key Findings & Model Internals")

tab1, tab2, tab3 = st.tabs(["PCA Visualization ", "Chi-Square Proof ", "Bayesian Analysis "])

with tab1:
    st.subheader(f"PCA Visualization ({N_DIMENSIONS}D → 2D)")
    st.markdown("Visual representation of how **Linear Algebra** (PCA) maps the high-dimensional data, clearly separating the Normal (Blue) and Anomalous (Red X) clusters.")

    if normal_data_2d is not None and anomalous_data_2d is not None:
        fig_pca, ax_pca = plt.subplots(figsize=(10, 6))
        ax_pca.scatter(normal_data_2d[:, 0], normal_data_2d[:, 1], c='blue', label='Normal Prompts', alpha=0.3, s=50)
        ax_pca.scatter(anomalous_data_2d[:, 0], anomalous_data_2d[:, 1], c='red', label='Anomalous Prompts', alpha=1.0, s=70, edgecolors='k', marker='X')
        ax_pca.set_title(f'PCA Visualization (Embedding Space)')
        ax_pca.set_xlabel('Principal Component 1')
        ax_pca.set_ylabel('Principal Component 2')
        ax_pca.legend()
        ax_pca.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig_pca)
    else:
        st.warning("PCA data could not be loaded for plotting.")


with tab2:
    st.subheader("Proof of Chi-Square (χ²) Assumption")
    st.markdown(f"""
    This chart validates the **Probability** assumption (Task 2). The squared Mahalanobis distance ($D^2$) for multivariate data should follow a Chi-Square distribution with **df={N_DIMENSIONS}**.
    """)

    if dists_normal_sq is not None:
        fig_chi, ax_chi = plt.subplots(figsize=(10, 6))
        ax_chi.hist(dists_normal_sq, bins=50, density=True, label='Actual Normal Data Distances', alpha=0.7)

        df = N_DIMENSIONS
        x_max = np.percentile(dists_normal_sq, 99.9) * 1.2 if dists_normal_sq else df * 2
        x = np.linspace(0, x_max, 300)
        pdf = chi2.pdf(x, df)
        ax_chi.plot(x, pdf, 'r-', lw=2, label=f'Theoretical Chi-Square PDF (df={df})')

        threshold_value = chi2.ppf(1.0 - P_VALUE_THRESHOLD, df)
        ax_chi.axvline(threshold_value, color='k', linestyle='--', lw=1.5, label=f'Anomaly Threshold (D² = {threshold_value:.2f})')

        ax_chi.set_title('Validation: Mahalanobis Distances vs. Chi-Square PDF')
        ax_chi.set_xlabel('Squared Mahalanobis Distance (D²)')
        ax_chi.set_ylabel('Probability Density')
        ax_chi.legend()
        ax_chi.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig_chi)
    else:
        st.warning("Could not calculate or load distances for normal data. Cannot generate Chi-Square plot.")


with tab3:
    st.subheader("Bayesian Analysis: The 'False Positive Paradox'")
    st.markdown(f"""
    It computes the posterior probability a flagged prompt is truly malicious.

    Using Bayes' Theorem with a **0.5% Prior Anomaly Rate** and our **1% False Positive Rate** (for simple calculation): The probability that a flagged prompt is truly malicious is only **$\sim 31.1\%$**.

    **Security/Governance Implication:** This result mandates a **tiered defense system**. The statistical filter must be a preliminary risk-scorer, passing alerts to a more accurate (but costlier) LLM or human analyst for final verification.
    """)
    st.code("""
    P(Malicious | Flagged) ≈ 31.1%
    """, language="python")

st.divider()

# ====================================================================
# 3. SECURITY & MLOPS DEMO (Implemented Defense)
# ====================================================================
st.header("3. 🛡️ Security & MLOps: Attack & Defense Demo")
st.markdown("This section showcases the implemented defense against **Adversarial Model Poisoning** (Task 5), aligning with **MLOps** principles for continuous model integrity.")

col_summary, col_drift_plot = st.columns([1, 2])

with col_summary:
    st.subheader("Automated Defense Summary (Cell 13)")
    st.markdown("The **`ModelDriftMonitor`** is the MLOps component that validates a newly retrained model before deployment.")
    
    # --- Hardcoded Values from your final output ---
    BASELINE_MEAN_NORM = 0.3023
    ATTACK_MEAN_DRIFT = 0.5705
    DRIFT_THRESHOLD = 0.5
    MONITOR_STATUS = "POISONING_DETECTED_OR_UNSTABLE"
    
    st.info("Initial Clean Model ($\mu_{clean}$ Norm)")
    st.metric(label="Baseline L2 Norm", value=f"{BASELINE_MEAN_NORM:.4f}")
    
    st.error("Poisoned Model Drift ($\mu_{poisoned}$)")
    st.metric(label="Detected Mean Drift (L2 Norm)", value=f"{ATTACK_MEAN_DRIFT:.4f}")
    
    if MONITOR_STATUS == "POISONING_DETECTED_OR_UNSTABLE":
        st.error(f"**🔴 DEPLOYMENT STATUS: REJECTED**")
    st.caption(f"Reason: Mean Drift ({ATTACK_MEAN_DRIFT:.4f}) exceeded MLOps threshold ({DRIFT_THRESHOLD}).")

with col_drift_plot:
    st.subheader("📈 Model Drift History Visualization")
    st.markdown("The graph below visually demonstrates the attack's progress and the exact point where the MLOps system triggers REJECTION.")

    if drift_history is not None:
        fig_drift, ax = plt.subplots(figsize=(10, 5))
        
        # CORRECTED: X-axis is cycles (1 to N), Y-axis is the list of floats
        cycles = range(1, len(drift_history) + 1)
        
        # Plot Mean Drift
        ax.plot(cycles, drift_history, 
                label='Mean Drift (L2 Norm)', color='#DC3545', linewidth=3)
        
        # Plot the MLOps Threshold line
        ax.axhline(DRIFT_THRESHOLD, color='#17A2B8', linestyle='--', 
                   label=f'MLOps Safety Threshold ({DRIFT_THRESHOLD})')
        
        # Highlight the area where the model is unsafe
        ax.fill_between(cycles, DRIFT_THRESHOLD, drift_history,
                        where=[d > DRIFT_THRESHOLD for d in drift_history],
                        color='#DC3545', alpha=0.3, label='Poisoning Detected Zone')

        ax.set_title('Model Mean Vector Drift Over Time (MLOps Defense Check)')
        ax.set_xlabel('Retraining Cycle Number')
        ax.set_ylabel('Mean Drift (L2 Norm)')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig_drift)
    else:
        st.error("Drift history file not found. Cannot generate visualization.")


st.divider()
st.caption("End of Spectra AI Prototype")