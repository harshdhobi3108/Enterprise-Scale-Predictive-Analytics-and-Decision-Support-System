"""
Retention SHAP Explainer
(Final Production Version | No Errors | Streamlit Safe)
"""

import joblib
import pandas as pd
import numpy as np

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

        self.preprocessor = None
        self.model = None

        # ======================================================
        # SAFE PIPELINE EXTRACTION
        # ======================================================
        if hasattr(self.pipeline, "named_steps"):

            for name, step in self.pipeline.named_steps.items():

                step_type = str(type(step)).lower()

                if "columntransformer" in step_type:
                    self.preprocessor = step

                elif "classifier" in name or "forest" in step_type:
                    self.model = step

        else:
            raise ValueError("Loaded model is not a Pipeline")

        if self.model is None:
            raise ValueError("Classifier not found in pipeline")

        # ======================================================
        # INIT SHAP
        # ======================================================
        if SHAP_AVAILABLE:
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception:
                self.explainer = None
        else:
            self.explainer = None

    # ==========================================================
    # HELPERS
    # ==========================================================
    def _transform(self, X):
        if self.preprocessor is not None:
            return self.preprocessor.transform(X)
        return X

    def _get_feature_names(self):
        try:
            return self.preprocessor.get_feature_names_out()
        except Exception:
            return None

    # ==========================================================
    # INSTANCE EXPLANATION
    # ==========================================================
    def explain_instance(self, X):

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
            print("Global SHAP error:", str(e))
            return None, X

    # ==========================================================
    # WATERFALL PLOT (FIXED FOR CLASSIFICATION)
    # ==========================================================
    def plot_waterfall(self, shap_values, X_named, index=0):

        if not SHAP_AVAILABLE or self.explainer is None:
            return None

        try:
            # ✅ FIX: use class 1
            expected_value = self.explainer.expected_value[1]
            shap_val = shap_values[1][index]
            features = X_named.iloc[index]

            fig = shap.plots._waterfall.waterfall_legacy(
                expected_value,
                shap_val,
                features
            )

            return fig

        except Exception as e:
            print("Waterfall error:", str(e))
            return None

    # ==========================================================
    # GLOBAL FEATURE IMPORTANCE
    # ==========================================================
    def global_importance(self, shap_values, X_named):

        if shap_values is None:
            return None

        try:
            importance = np.abs(shap_values[1]).mean(axis=0)

            df = pd.DataFrame({
                "feature": X_named.columns,
                "importance": importance
            }).sort_values(by="importance", ascending=False)

            return df

        except Exception as e:
            print("Importance error:", str(e))
            return None