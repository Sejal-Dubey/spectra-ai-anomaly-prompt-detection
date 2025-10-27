# Spectra AI Mini Challenge: 

## Project Overview 🛡️

This project implements a prototype designed to detect anomalous and potentially malicious prompts submitted to a Large Language Model (LLM). The core detection mechanism relies on a sophisticated statistical model that flags inputs that deviate from a learned baseline of "normal" user behavior.

The solution demonstrates expertise in **Linear Algebra**, **Probability**, **Bayesian Analysis**, and **MLOps Security**.

## Core Methodology

The system operates on the principle of **Deviation Detection in High-Dimensional Space** (384-dimensions):

1.  **Modeling Normalcy (Task 1 - Linear Algebra):** The system trains on a diverse set of benign prompts to calculate the **Mean Vector ($\boldsymbol{\mu}$)** and the **Inverse Covariance Matrix ($\mathbf{\Sigma}^{-1}$)**, which define the center and complex shape of the "normal" prompt cluster.
2.  **Anomaly Scoring (Task 1/2 - Mahalanobis Distance):** A new prompt's embedding ($\mathbf{x}$) is measured using the **Mahalanobis Distance ($\mathbf{D}^2$)**. This normalized distance accounts for the correlation between all 384 dimensions, providing a robust measure of statistical anomaly.
3.  **Risk Quantification (Task 2 - Probability):** The $\mathbf{D}^2$ is converted into a **P-Value** via a **Chi-Square ($\mathbf{\chi}^2$) Test**, which quantifies the probability that the prompt belongs to the normal distribution.
4.  **Proactive Defense (Task 5 - MLOps Security):** A **Model Drift Monitor** is implemented to automatically detect attempts to poison the statistical baseline, ensuring the detector's integrity over time.

## Key Deliverables & Code Structure

The project is organized into a clean, reusable structure:

| File/Directory | Description | Alignment with Tasks |
| :--- | :--- | :--- |
| `main_notebook.ipynb` | **Primary Documentation.** Contains all step-by-step mathematical calculations, data generation, visualizations (PCA, Chi-Square, Drift Graph), and the security simulation walk-through. | Tasks 1, 2, 3, 4, 5 |
| `src/anomaly_detector.py` | **Core Implementation.** A clean, production-style Python class containing the Mahalanobis distance logic, Chi-Square test, and anomaly flagging. | Tasks 1 & 2 (Technical Impl.) |
| `app.py` | **Interactive Prototype.** A Streamlit application that provides a real-time demo and visual documentation of all key findings. | Task 4 (Visualization) |
| `drift_history.pkl` | **Model Artifact.** Stores the data used to prove the MLOps security defense. | Task 5 (Security/MLOps) |

## Deployment & Production Considerations (MLOps Focus)

This solution is designed with scalability and integrity in mind:

### **Tiered Security Architecture (Governance)**

  * **Bayesian Justification (Task 3):** Since the Bayesian analysis showed that only $\sim 31\%$ of flags are truly malicious, the system is designed as a **Tiered Defense**:
    1.  **Level 1 (Statistical):** The current system provides **fast, low-cost screening**.
    2.  **Level 2 (Fallback):** Prompts flagged as ANOMALOUS are passed to a secondary, more accurate (but expensive) LLM-based intent analyzer or human review.

### **Proactive Model Integrity (MLOps)**

  * **The Threat:** The system is vulnerable to **Adversarial Model Poisoning** during routine retraining.
  * **The Defense (Implemented):** The **`ModelDriftMonitor`** tracks the **Mean Drift** (L2 norm) of the detector's statistical center ($\boldsymbol{\mu}$) after every update. If the drift exceeds the safety threshold ($\mathbf{0.5}$), the model is automatically **REJECTED** from deployment. This proactive check ensures the continuous integrity of the security tool.

## Setup and Run Instructions

1.  **Clone Repository:**
    ```bash
    git clone [Your Repository URL]
    cd spectra-ai-challenge
    ```
2.  **Setup Environment:**
    ```bash
    # Create and activate virtual environment
    python -m venv venv
    source venv/bin/activate  # On Linux/Mac
    .\venv\Scripts\activate   # On Windows

    # Install dependencies
    pip install -r requirements.txt
    ```
3.  **Run Notebook (Generate Models):**
      * Open `main_notebook.ipynb` in Jupyter or VS Code.
      * Run **all cells** sequentially to train the detector, run the security simulation, and generate the necessary `.pkl` files (`trained_detector.pkl`, `drift_history.pkl`, etc.).
4.  **Launch Demo UI (View Results):**
    ```bash
    streamlit run app.py
    ```
