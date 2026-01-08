"""
RUL Model Training Pipeline
============================

Complete pipeline for training battery RUL prediction models.
Handles data loading, feature engineering, model training, and evaluation.
"""

import sqlite3
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from backend.ml.feature_engineering import BatteryFeatureEngineer
from backend.ml.models.xgboost_model import XGBoostRULModel


class RULTrainingPipeline:
    """
    Complete training pipeline for RUL models.

    Usage:
        pipeline = RULTrainingPipeline(model_type='xgboost')
        metrics = pipeline.train_from_database(
            db_path='nasa_amperedata_full.db',
            battery_ids=['B0005', 'B0006', 'B0007']
        )
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        test_size: float = 0.2,
        random_state: int = 42,
        eol_threshold: float = 0.8,
    ):
        """
        Initialize training pipeline.

        Args:
            model_type: Type of model ('xgboost', 'random_forest', etc.)
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
            eol_threshold: End-of-life threshold (fraction of initial capacity)
        """
        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state
        self.eol_threshold = eol_threshold

        # Components
        self.feature_engineer = BatteryFeatureEngineer(eol_threshold=eol_threshold)
        self.scaler = StandardScaler()
        self.model = None

        # Training data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

    def load_battery_data_from_db(
        self, db_path: str, battery_ids: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Load battery data from SQLite database.

        Args:
            db_path: Path to database
            battery_ids: List of battery IDs to load (None = all)

        Returns:
            DataFrame with battery cycle data
        """
        conn = sqlite3.connect(db_path)

        # Build query - Join capacity_fade with cycles for complete data
        if battery_ids:
            battery_filter = f"WHERE cf.battery_id IN ({','.join(['?'] * len(battery_ids))})"
            query = f"""
                SELECT
                    cf.battery_id,
                    cf.cycle_number,
                    cf.capacity_ah as discharge_capacity,
                    c.voltage,
                    c.current,
                    c.time_seconds
                FROM capacity_fade cf
                LEFT JOIN cycles c ON cf.battery_id = c.battery_id AND cf.test_id = c.test_id
                {battery_filter}
                ORDER BY cf.battery_id, cf.cycle_number, c.time_seconds
            """
            df = pd.read_sql_query(query, conn, params=battery_ids)
        else:
            query = """
                SELECT
                    cf.battery_id,
                    cf.cycle_number,
                    cf.capacity_ah as discharge_capacity,
                    c.voltage,
                    c.current,
                    c.time_seconds
                FROM capacity_fade cf
                LEFT JOIN cycles c ON cf.battery_id = c.battery_id AND cf.test_id = c.test_id
                ORDER BY cf.battery_id, cf.cycle_number, c.time_seconds
            """
            df = pd.read_sql_query(query, conn)

        # Add charge capacity (estimate if not available)
        if "charge_capacity" not in df.columns:
            df["charge_capacity"] = df["discharge_capacity"] * 1.02  # Typical CE ~98%

        conn.close()

        return df

    def extract_features_from_batteries(
        self, battery_data: pd.DataFrame, include_temperature: bool = False
    ) -> pd.DataFrame:
        """
        Extract features from all batteries in dataset.

        Args:
            battery_data: Raw battery cycle data
            include_temperature: Include temperature features

        Returns:
            DataFrame with extracted features
        """
        all_features = []

        for battery_id in battery_data["battery_id"].unique():
            print(f"Extracting features from {battery_id}...")

            battery_subset = battery_data[battery_data["battery_id"] == battery_id].copy()

            features = self.feature_engineer.extract_features(
                battery_subset, include_temperature=include_temperature
            )

            all_features.append(features)

        combined_features = pd.concat(all_features, ignore_index=True)

        print(
            f"✅ Extracted {len(combined_features)} feature sets from {battery_data['battery_id'].nunique()} batteries"
        )

        return combined_features

    def prepare_training_data(
        self, features_df: pd.DataFrame, scale_features: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training (split and scale).

        Args:
            features_df: DataFrame with extracted features
            scale_features: Whether to scale features

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Separate features and target
        feature_cols = [
            col for col in features_df.columns if col not in ["battery_id", "cycle_number", "RUL"]
        ]

        X = features_df[feature_cols].copy()
        y = features_df["RUL"].copy()

        # Handle any missing values
        X = X.fillna(0)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, shuffle=True
        )

        # Scale features
        if scale_features:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        self.feature_names = feature_cols
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        print(f"✅ Training set: {len(X_train)} samples")
        print(f"✅ Test set: {len(X_test)} samples")
        print(f"✅ Features: {len(feature_cols)}")

        return X_train, X_test, y_train, y_test

    def train_model(
        self,
        hyperparameters: Optional[dict[str, Any]] = None,
        early_stopping: bool = True,
    ) -> dict[str, float]:
        """
        Train the model.

        Args:
            hyperparameters: Optional hyperparameters to override defaults
            early_stopping: Use early stopping with validation set

        Returns:
            Dict of evaluation metrics
        """
        if self.X_train is None:
            raise ValueError("No training data prepared. Call prepare_training_data() first.")

        print(f"\n🤖 Training {self.model_type} model...")

        # Initialize model
        if self.model_type == "xgboost":
            self.model = XGBoostRULModel(hyperparameters=hyperparameters)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Train
        if early_stopping:
            self.model.train(self.X_train, self.y_train, self.X_test, self.y_test, verbose=False)
        else:
            self.model.train(self.X_train, self.y_train, verbose=False)

        # Evaluate
        metrics = self.evaluate_model()

        print("\n✅ Training complete!")
        print(f"   MAE: {metrics['test_mae']:.2f} cycles")
        print(f"   RMSE: {metrics['test_rmse']:.2f} cycles")
        print(f"   R²: {metrics['test_r2']:.4f}")

        return metrics

    def evaluate_model(self) -> dict[str, float]:
        """
        Evaluate model performance.

        Returns:
            Dict of evaluation metrics
        """
        if self.model is None or not self.model.is_trained:
            raise ValueError("Model not trained yet.")

        # Training metrics
        y_train_pred = self.model.predict(self.X_train)
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        train_r2 = r2_score(self.y_train, y_train_pred)

        # Test metrics
        y_test_pred = self.model.predict(self.X_test)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        test_r2 = r2_score(self.y_test, y_test_pred)

        # MAPE (Mean Absolute Percentage Error)
        test_mape = np.mean(np.abs((self.y_test - y_test_pred) / (self.y_test + 1))) * 100

        return {
            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "train_r2": train_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "test_mape": test_mape,
        }

    def train_from_database(
        self,
        db_path: str,
        battery_ids: Optional[list[str]] = None,
        hyperparameters: Optional[dict[str, Any]] = None,
    ) -> dict[str, float]:
        """
        Complete training pipeline from database.

        Args:
            db_path: Path to SQLite database
            battery_ids: List of battery IDs to train on (None = all)
            hyperparameters: Optional model hyperparameters

        Returns:
            Dict of evaluation metrics
        """
        print("🚀 Starting RUL training pipeline...")
        print(f"   Database: {db_path}")
        print(f"   Batteries: {battery_ids if battery_ids else 'All'}")

        # 1. Load data
        print("\n📊 Loading battery data...")
        battery_data = self.load_battery_data_from_db(db_path, battery_ids)

        # 2. Extract features
        print("\n🔧 Extracting features...")
        features_df = self.extract_features_from_batteries(battery_data)

        # 3. Prepare data
        print("\n📐 Preparing training data...")
        self.prepare_training_data(features_df)

        # 4. Train model
        metrics = self.train_model(hyperparameters=hyperparameters)

        return metrics

    def save_pipeline(self, output_dir: str):
        """
        Save complete pipeline (model + scaler).

        Args:
            output_dir: Directory to save pipeline components
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = output_path / f"rul_model_{self.model_type}.joblib"
        self.model.save(str(model_path))
        print(f"✅ Model saved: {model_path}")

        # Save scaler
        scaler_path = output_path / "feature_scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ Scaler saved: {scaler_path}")

        # Save feature names
        features_path = output_path / "feature_names.txt"
        with open(features_path, "w") as f:
            f.write("\n".join(self.feature_names))
        print(f"✅ Features saved: {features_path}")

        # Save metadata
        metadata = {
            "model_type": self.model_type,
            "test_size": self.test_size,
            "eol_threshold": self.eol_threshold,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
        }
        metadata_path = output_path / "pipeline_metadata.joblib"
        joblib.dump(metadata, metadata_path)
        print(f"✅ Metadata saved: {metadata_path}")

    def load_pipeline(self, input_dir: str):
        """
        Load complete pipeline from disk.

        Args:
            input_dir: Directory containing pipeline components
        """
        input_path = Path(input_dir)

        # Load metadata
        metadata_path = input_path / "pipeline_metadata.joblib"
        metadata = joblib.load(metadata_path)

        self.model_type = metadata["model_type"]
        self.feature_names = metadata["feature_names"]

        # Load model
        model_path = input_path / f"rul_model_{self.model_type}.joblib"
        if self.model_type == "xgboost":
            self.model = XGBoostRULModel()
        self.model.load(str(model_path))

        # Load scaler
        scaler_path = input_path / "feature_scaler.joblib"
        self.scaler = joblib.load(scaler_path)

        print(f"✅ Pipeline loaded from {input_path}")
