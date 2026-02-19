"""
Retention SHAP Explainer
With Proper Feature Names
"""

import shap
import joblib
import pandas as pd


class RetentionExplainer:

    def __init__(self, model_path: str):

        self.pipeline = joblib.load(model_path)

        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.model = self.pipeline.named_steps["classifier"]

        self.explainer = shap.TreeExplainer(self.model)

    def explain_instance(self, X):

        X_transformed = self.preprocessor.transform(X)

        shap_values = self.explainer.shap_values(X_transformed)

        feature_names = X.columns.tolist()

        X_named = pd.DataFrame(
            X_transformed,
            columns=feature_names
        )

        return shap_values, X_named

    def explain_global(self, X):

        X_transformed = self.preprocessor.transform(X)

        shap_values = self.explainer.shap_values(X_transformed)

        feature_names = X.columns.tolist()

        X_named = pd.DataFrame(
            X_transformed,
            columns=feature_names
        )

        return shap_values, X_named
