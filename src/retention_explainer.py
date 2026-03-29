"""
Retention SHAP Explainer
(Final Stable Version - No Errors)
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

        # Extract pipeline safely
        if hasattr(self.pipeline, "named_steps"):

            for name, step in self.pipeline.named_steps.items():

                if "columntransformer" in str(type(step)).lower():
                    self.preprocessor = step

                else:
                    self.model = step

        if self.model is None:
            raise ValueError("Model not found in pipeline")

        # Init SHAP
        if SHAP_AVAILABLE:
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except:
                self.explainer = None
        else:
            self.explainer = None

    # -----------------------------
    # Transform
    # -----------------------------
    def _transform(self, X):
        if self.preprocessor:
            return self.preprocessor.transform(X)
        return X

    def _feature_names(self):
        try:
            return self.preprocessor.get_feature_names_out()
        except:
            return None

    # -----------------------------
    # Instance SHAP
    # -----------------------------
    def explain_instance(self, X):

        if not SHAP_AVAILABLE or self.explainer is None:
            return None, X

        X_t = self._transform(X)
        shap_values = self.explainer.shap_values(X_t)

        names = self._feature_names()
        if names is None:
            names = [f"f_{i}" for i in range(X_t.shape[1])]

        X_named = pd.DataFrame(X_t, columns=names)

        return shap_values, X_named

    # -----------------------------
    # Global SHAP
    # -----------------------------
    def explain_global(self, X):

        if not SHAP_AVAILABLE or self.explainer is None:
            return None, X

        X_t = self._transform(X)
        shap_values = self.explainer.shap_values(X_t)

        names = self._feature_names()
        if names is None:
            names = [f"f_{i}" for i in range(X_t.shape[1])]

        X_named = pd.DataFrame(X_t, columns=names)

        return shap_values, X_named