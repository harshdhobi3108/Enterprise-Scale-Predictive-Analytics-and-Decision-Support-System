"""
Retention SHAP Explainer
(Final Stable Version - Streamlit + Raw Model Safe)
"""

import joblib
import pandas as pd
import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False


class RetentionExplainer:

    def __init__(self, model_path: str):

        self.pipeline = joblib.load(model_path)

        self.preprocessor = None
        self.model = None

        # ==========================================================
        # 🔥 HANDLE BOTH CASES: PIPELINE + RAW MODEL
        # ==========================================================
        if hasattr(self.pipeline, "named_steps"):
            # Pipeline case
            for name, step in self.pipeline.named_steps.items():

                # Detect preprocessor
                if "columntransformer" in str(type(step)).lower():
                    self.preprocessor = step

                # Assume last step is model
                self.model = step

        else:
            # 🔥 RAW MODEL CASE (YOUR CASE)
            self.model = self.pipeline
            self.preprocessor = None

        if self.model is None:
            raise ValueError("Model could not be loaded")

        # ==========================================================
        # SHAP INIT (SAFE)
        # ==========================================================
        if SHAP_AVAILABLE:
            try:
                # Try TreeExplainer first (best for tree models)
                self.explainer = shap.TreeExplainer(self.model)
            except Exception:
                try:
                    # Fallback (universal but slower)
                    self.explainer = shap.Explainer(self.model)
                except Exception:
                    self.explainer = None
        else:
            self.explainer = None

    # ==========================================================
    # TRANSFORM (SAFE)
    # ==========================================================
    def _transform(self, X):
        try:
            if self.preprocessor is not None:
                return self.preprocessor.transform(X)
            return X
        except Exception:
            return X  # fallback

    def _feature_names(self, X, X_t):
        try:
            if self.preprocessor is not None:
                return self.preprocessor.get_feature_names_out()
            else:
                return X.columns.tolist()
        except Exception:
            return [f"f_{i}" for i in range(X_t.shape[1])]

    # ==========================================================
    # INSTANCE SHAP
    # ==========================================================
    def explain_instance(self, X):

        if not SHAP_AVAILABLE or self.explainer is None:
            return None, X

        try:
            X_t = self._transform(X)

            # SHAP values
            shap_values = self.explainer.shap_values(X_t)

            # Feature names
            names = self._feature_names(X, X_t)

            X_named = pd.DataFrame(X_t, columns=names)

            return shap_values, X_named

        except Exception:
            return None, X

    # ==========================================================
    # GLOBAL SHAP
    # ==========================================================
    def explain_global(self, X):

        if not SHAP_AVAILABLE or self.explainer is None:
            return None, X

        try:
            X_t = self._transform(X)

            shap_values = self.explainer.shap_values(X_t)

            names = self._feature_names(X, X_t)

            X_named = pd.DataFrame(X_t, columns=names)

            return shap_values, X_named

        except Exception:
            return None, X