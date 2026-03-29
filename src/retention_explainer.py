"""
Retention SHAP Explainer
(Deployment-Safe Version - SHAP Optional)
"""

import joblib
import pandas as pd

# ==========================================================
# SAFE SHAP IMPORT
# ==========================================================
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False


class RetentionExplainer:

    def __init__(self, model_path: str):

        self.pipeline = joblib.load(model_path)

        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.model = self.pipeline.named_steps["classifier"]

        # Initialize explainer only if SHAP is available
        if SHAP_AVAILABLE:
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception:
                self.explainer = None
        else:
            self.explainer = None

    # ==========================================================
    # LOCAL EXPLANATION
    # ==========================================================
    def explain_instance(self, X):

        # If SHAP not available → safe fallback
        if not SHAP_AVAILABLE or self.explainer is None:
            return None, X

        try:
            X_transformed = self.preprocessor.transform(X)

            shap_values = self.explainer.shap_values(X_transformed)

            feature_names = X.columns.tolist()

            X_named = pd.DataFrame(
                X_transformed,
                columns=feature_names
            )

            return shap_values, X_named

        except Exception:
            return None, X

    # ==========================================================
    # GLOBAL EXPLANATION
    # ==========================================================
    def explain_global(self, X):

        # If SHAP not available → safe fallback
        if not SHAP_AVAILABLE or self.explainer is None:
            return None, X

        try:
            X_transformed = self.preprocessor.transform(X)

            shap_values = self.explainer.shap_values(X_transformed)

            feature_names = X.columns.tolist()

            X_named = pd.DataFrame(
                X_transformed,
                columns=feature_names
            )

            return shap_values, X_named

        except Exception:
            return None, X