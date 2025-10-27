# src/anomaly_detector.py

import numpy as np
from scipy.stats import chi2

class PromptAnomalyDetector:
    """
    A statistical guardrail designed to detect anomalous prompts by measuring 
    their Mahalanobis distance from a learned "normal" prompt distribution.
    
    This class performs the core Linear Algebra and Probability tasks required.
    """
    
    def __init__(self, dimensions):
        """
        Initializes the detector by setting the expected dimensionality of the embeddings.
        
        Args:
            dimensions (int): The degrees of freedom (df) for the Chi-Square test, 
                              equal to the dimension of the embedding vectors (e.g., 384).
        """
        self.dimensions = dimensions
        # These matrices define the learned statistical "fingerprint" of normal behavior
        self.mean_vector = None
        self.inv_cov_matrix = None
        print(f"Detector initialized for {dimensions}-dimensional embeddings.")

    def fit(self, normal_embeddings):
        """
        [ Linear Algebra: Compute Covariance Matrix]
        Learns the statistical parameters (center and shape) of the normal distribution.

        Args:
            normal_embeddings (np.array): A [n_samples, n_dimensions] array
                                           of "normal" prompt embeddings.
        """
        print("Fitting normal distribution baseline (calculating mean and covariance)...")
        
        # 1. Calculate the Mean Vector (mu - μ): The central point of the normal cluster.
        self.mean_vector = np.mean(normal_embeddings, axis=0)
        
        # 2. Calculate the Covariance Matrix (Sigma - Σ):
        # This describes the statistical shape, spread, and correlation between the 384 dimensions.
        cov_matrix = np.cov(normal_embeddings, rowvar=False)
        
        # 3. Calculate the Inverse Covariance Matrix (Sigma^-1 - Σ⁻¹):
        # This is the "Mahalanobis ruler" used to normalize distance relative to the cluster's shape.
        # --- PRODUCTION/LINEAR ALGEBRA DETAIL ---
        # We add a small regularization term (ε*I) for **numerical stability**.
        # This ensures the matrix is non-singular (invertible), preventing runtime errors
        # in high-dimensional or highly correlated data.
        reg = 1e-6 
        self.inv_cov_matrix = np.linalg.inv(cov_matrix + np.eye(self.dimensions) * reg)
        
        print(f"Fit complete. Learned {len(normal_embeddings)} samples.")

    def _compute_mahalanobis_sq(self, new_embedding):
        """
        [Linear Algebra: Compute Mahalanobis Distance]
        Calculates the squared Mahalanobis distance ($D^2$) for a new embedding.
        
        The result is a normalized measure of distance, robust to feature scale and correlation.
        
        Formula: $D^2 = (\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$
        """
        if self.mean_vector is None or self.inv_cov_matrix is None:
            raise ValueError("Model has not been fit yet. Call .fit() first.")
        
        # Vector difference: (x - μ)
        diff = new_embedding - self.mean_vector
        
        # Matrix multiplication: (x - μ)ᵀ * Σ⁻¹ * (x - μ)
        # The dot product handles the multiple matrix multiplications efficiently.
        dist_sq = diff.T @ self.inv_cov_matrix @ diff
        
        return dist_sq

    def compute_p_value(self, new_embedding):
        """
        [ Probability: Compute Probability/P-Value]
        Converts the Mahalanobis distance into a probability.
        
        The probability quantifies how likely the new prompt is to belong to the 
        established normal distribution.
        
        Returns:
            (float, float): The p-value and the squared Mahalanobis distance.
        """
        # Calculate the squared distance.
        mahal_sq_dist = self._compute_mahalanobis_sq(new_embedding)
        
        # The Mahalanobis distance of multivariate normal data is distributed as 
        # a Chi-Square distribution (χ²) with degrees of freedom (df) equal to 
        # the number of dimensions (384).
        
        # Use the Survival Function (1 - CDF) to get the p-value:
        # p-value = P(χ² > D²). A small p-value indicates a high degree of anomaly.
        p_value = chi2.sf(mahal_sq_dist, df=self.dimensions)
        return p_value, mahal_sq_dist

    def predict(self, new_embedding, p_value_threshold=0.01):
        """
        Flags a new prompt based on the calculated p-value and a risk threshold.
        
        Args:
            new_embedding (np.array): The prompt's embedding vector.
            p_value_threshold (float): The maximum acceptable probability 
                                       for a prompt to be considered 'normal'.
                                       (e.g., 0.01 means accepting a 1% chance of False Alarm).
                                       
        Returns:
            (str, float, float): ('ANOMALOUS'/'NORMAL', p-value, distance)
        """
        p_value, mahal_sq_dist = self.compute_p_value(new_embedding)
        
        # Flagging Logic: If the probability of being normal is too low (p-value < threshold), flag it.
        if p_value < p_value_threshold:
            return "ANOMALOUS", p_value, mahal_sq_dist
        else:
            return "NORMAL", p_value, mahal_sq_dist