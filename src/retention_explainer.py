"""
Retention SHAP Explainer
(Production-Grade | Fully Safe | Streamlit Compatible)
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

        # Load model
        self.pipeline = joblib.load(model_path)

        self.preprocessor = None
        self.model = None

        # ======================================================
        # SAFE PIPELINE EXTRACTION (NO HARD-CODING)
        # ======================================================
        if hasattr(self.pipeline, "named_steps"):

            for name, step in self.pipeline.named_steps.items():

                step_type = str(type(step)).lower()

                if "columntransformer" in step_type:
                    self.preprocessor = step

                elif "classifier" in name or "forest" in step_type:
                    self.model = step

        else:
            raise ValueError("Loaded model is not a valid sklearn Pipeline")

        if self.model is None:
            raise ValueError("No classifier found inside pipeline")

        # ======================================================
        # INIT SHAP (SAFE)
        # ======================================================
        if SHAP_AVAILABLE:
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception:
                self.explainer = None
        else:
            self.explainer = None

    # ==========================================================
    # GET FEATURE NAMES AFTER TRANSFORMATION
    # ==========================================================
    def _get_feature_names(self):

        try:
            return self.preprocessor.get_feature_names_out()
        except Exception:
            return None

    # ==========================================================
    # TRANSFORM DATA SAFELY
    # ==========================================================
    def _transform(self, X):

        if self.preprocessor is not None:
            return self.preprocessor.transform(X)
        return X

    # ==========================================================
    # LOCAL EXPLANATION
    # ==========================================================
    def explain_instance(self, X):

        if not SHAP_AVAILABLE or self.explainer is None:
            return None, X

        try:
            X_transformed = self._transform(X)

            shap_values = self.explainer.shap_values(X_transformed)

            feature_names = self._get_feature_names()

            # Fallback if feature names not available
            if feature_names is None:
                feature_names = [f"f_{i}" for i in range(X_transformed.shape[1])]

            X_named = pd.DataFrame(
                X_transformed,
                columns=feature_names
            )

            return shap_values, X_named

        except Exception as e:
            print("SHAP instance error:", str(e))
            return None, X

    # ==========================================================
    # GLOBAL EXPLANATION
    # ==========================================================
    def explain_global(self, X):

        if not SHAP_AVAILABLE or self.explainer is None:
            return None, X

        try:
            X_transformed = self._transform(X)

            shap_values = self.explainer.shap_values(X_transformed)

            feature_names = self._get_feature_names()

            if feature_names is None:
                feature_names = [f"f_{i}" for i in range(X_transformed.shape[1])]

            X_named = pd.DataFrame(
                X_transformed,
                columns=feature_names
            )

            return shap_values, X_named

        except Exception as e:
            print("SHAP global error:", str(e))
            return None, X