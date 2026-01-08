"""
Base Model Interface for RUL Prediction
========================================

Abstract base class that all RUL prediction models must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Dict, Tuple

import joblib
import numpy as np
import pandas as pd


class BaseRULModel(ABC):
    """Abstract base class for RUL prediction models."""

    def __init__(self, model_name: str, hyperparameters: Optional[Dict[str, Any]] = None):
        """
        Initialize base model.

        Args:
            model_name: Name/identifier for the model
            hyperparameters: Dict of hyperparameters
        """
        self.model_name = model_name
        self.hyperparameters = hyperparameters or self.get_default_hyperparameters()
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.training_history = []

    @abstractmethod
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters for the model."""

    @abstractmethod
    def _build_model(self):
        """Build the model with current hyperparameters."""

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "BaseRULModel":
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels (RUL values)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Self for method chaining
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features to predict on

        Returns:
            Predicted RUL values
        """

    def predict_with_uncertainty(
        self, X: pd.DataFrame, n_iterations: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates (if supported).

        Args:
            X: Features to predict on
            n_iterations: Number of bootstrap iterations

        Returns:
            Tuple of (predictions, uncertainties)
        """
        predictions = self.predict(X)
        uncertainties = np.zeros_like(predictions)  # Default: no uncertainty
        return predictions, uncertainties

    def save(self, path: str):
        """
        Save model to disk.

        Args:
            path: Path to save model
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "model_name": self.model_name,
            "hyperparameters": self.hyperparameters,
            "is_trained": self.is_trained,
            "feature_names": self.feature_names,
            "training_history": self.training_history,
        }

        joblib.dump(model_data, save_path)

    def load(self, path: str) -> "BaseRULModel":
        """
        Load model from disk.

        Args:
            path: Path to load model from

        Returns:
            Self for method chaining
        """
        model_data = joblib.load(path)

        self.model = model_data["model"]
        self.model_name = model_data["model_name"]
        self.hyperparameters = model_data["hyperparameters"]
        self.is_trained = model_data["is_trained"]
        self.feature_names = model_data["feature_names"]
        self.training_history = model_data.get("training_history", [])

        return self

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance (if supported by model).

        Returns:
            DataFrame with feature importance scores, or None
        """
        return None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(model_name='{self.model_name}', trained={self.is_trained})"
        )
