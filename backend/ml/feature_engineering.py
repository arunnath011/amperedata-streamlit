"""
Battery Feature Engineering for RUL Prediction
===============================================

Extracts 20+ features from time-series battery data for machine learning models.

Features include:
- Capacity metrics (fade rate, variance, trends)
- Voltage characteristics (mean, drop rate, variance)
- Current patterns (charge rate, variance)
- Efficiency metrics (coulombic, energy)
- Temperature profiles (if available)
- Statistical features (skewness, kurtosis, autocorrelation)
"""

import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")


class BatteryFeatureEngineer:
    """Extract features from battery time-series data for RUL prediction."""

    # Feature definitions
    CAPACITY_FEATURES = [
        "initial_capacity",
        "current_capacity",
        "capacity_fade_rate",
        "capacity_std",
        "capacity_variance",
        "capacity_range",
        "capacity_trend_strength",
    ]

    VOLTAGE_FEATURES = [
        "mean_voltage",
        "voltage_drop_rate",
        "voltage_variance",
        "voltage_range",
        "voltage_peak_count",
    ]

    CURRENT_FEATURES = [
        "mean_current",
        "current_variance",
        "charge_rate",
        "discharge_rate",
        "current_asymmetry",
    ]

    EFFICIENCY_FEATURES = [
        "coulombic_efficiency",
        "coulombic_efficiency_variance",
        "energy_efficiency",
        "power_fade_rate",
    ]

    TEMPERATURE_FEATURES = [
        "max_temperature",
        "avg_temperature",
        "temp_variance",
        "temp_rise_rate",
    ]

    STATISTICAL_FEATURES = [
        "capacity_skewness",
        "capacity_kurtosis",
        "voltage_autocorr",
        "capacity_entropy",
    ]

    CYCLE_FEATURES = [
        "cycle_count",
        "avg_cycle_time",
        "cycles_since_capacity_drop",
        "time_at_high_voltage",
        "time_at_high_current",
    ]

    def __init__(self, eol_threshold: float = 0.8):
        """
        Initialize feature engineer.

        Args:
            eol_threshold: End-of-life threshold (fraction of initial capacity)
        """
        self.eol_threshold = eol_threshold
        self.feature_names = (
            self.CAPACITY_FEATURES
            + self.VOLTAGE_FEATURES
            + self.CURRENT_FEATURES
            + self.EFFICIENCY_FEATURES
            + self.STATISTICAL_FEATURES
            + self.CYCLE_FEATURES
        )

    def extract_features(
        self, battery_data: pd.DataFrame, include_temperature: bool = False
    ) -> pd.DataFrame:
        """
        Extract all features from battery data.

        Args:
            battery_data: DataFrame with columns:
                         - battery_id
                         - cycle_number
                         - discharge_capacity
                         - charge_capacity (optional)
                         - voltage
                         - current
                         - time_seconds
                         - temperature (optional)
            include_temperature: Include temperature features if available

        Returns:
            DataFrame with features for each cycle
        """
        features_list = []

        battery_id = battery_data["battery_id"].iloc[0]
        max_cycle = battery_data["cycle_number"].max()

        # Extract features for each cycle (using data up to that cycle)
        for cycle in range(1, max_cycle + 1):
            # Get data up to current cycle
            cycle_data = battery_data[battery_data["cycle_number"] <= cycle].copy()

            if len(cycle_data) < 2:
                continue

            # Extract features
            feature_dict = {
                "battery_id": battery_id,
                "cycle_number": cycle,
            }

            # Capacity features
            feature_dict.update(self._extract_capacity_features(cycle_data))

            # Voltage features
            feature_dict.update(self._extract_voltage_features(cycle_data))

            # Current features
            feature_dict.update(self._extract_current_features(cycle_data))

            # Efficiency features
            if "charge_capacity" in cycle_data.columns:
                feature_dict.update(self._extract_efficiency_features(cycle_data))

            # Statistical features
            feature_dict.update(self._extract_statistical_features(cycle_data))

            # Cycle features
            feature_dict.update(self._extract_cycle_features(cycle_data))

            # Temperature features (optional)
            if include_temperature and "temperature" in cycle_data.columns:
                feature_dict.update(self._extract_temperature_features(cycle_data))

            # Calculate RUL (target variable)
            feature_dict["RUL"] = self._calculate_rul(battery_data, cycle)

            features_list.append(feature_dict)

        return pd.DataFrame(features_list)

    def _extract_capacity_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract capacity-related features."""
        capacity = data.groupby("cycle_number")["discharge_capacity"].mean()

        if len(capacity) < 2:
            return {feat: 0.0 for feat in self.CAPACITY_FEATURES}

        # Calculate fade rate (linear regression slope)
        x = np.arange(len(capacity))
        slope, intercept = np.polyfit(x, capacity.values, 1)

        # Trend strength (R²)
        y_pred = slope * x + intercept
        ss_res = np.sum((capacity.values - y_pred) ** 2)
        ss_tot = np.sum((capacity.values - np.mean(capacity.values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            "initial_capacity": capacity.iloc[0],
            "current_capacity": capacity.iloc[-1],
            "capacity_fade_rate": slope,
            "capacity_std": capacity.std(),
            "capacity_variance": capacity.var(),
            "capacity_range": capacity.max() - capacity.min(),
            "capacity_trend_strength": r_squared,
        }

    def _extract_voltage_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract voltage-related features."""
        if "voltage" not in data.columns:
            return {feat: 0.0 for feat in self.VOLTAGE_FEATURES}

        voltage = data["voltage"].dropna()

        if len(voltage) < 2:
            return {feat: 0.0 for feat in self.VOLTAGE_FEATURES}

        # Voltage drop rate
        v_by_cycle = data.groupby("cycle_number")["voltage"].mean()
        if len(v_by_cycle) > 1:
            x = np.arange(len(v_by_cycle))
            voltage_drop_rate, _ = np.polyfit(x, v_by_cycle.values, 1)
        else:
            voltage_drop_rate = 0

        # Peak count (voltage oscillations)
        try:
            peaks, _ = find_peaks(voltage.values, height=voltage.mean())
            peak_count = len(peaks) / len(voltage)  # Normalized
        except:
            peak_count = 0

        return {
            "mean_voltage": voltage.mean(),
            "voltage_drop_rate": voltage_drop_rate,
            "voltage_variance": voltage.var(),
            "voltage_range": voltage.max() - voltage.min(),
            "voltage_peak_count": peak_count,
        }

    def _extract_current_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract current-related features."""
        if "current" not in data.columns:
            return {feat: 0.0 for feat in self.CURRENT_FEATURES}

        current = data["current"].dropna()

        if len(current) < 2:
            return {feat: 0.0 for feat in self.CURRENT_FEATURES}

        # Separate charge and discharge
        charge_current = current[current > 0]
        discharge_current = current[current < 0]

        charge_rate = charge_current.mean() if len(charge_current) > 0 else 0
        discharge_rate = abs(discharge_current.mean()) if len(discharge_current) > 0 else 0

        # Current asymmetry (charge vs discharge)
        current_asymmetry = (charge_rate - discharge_rate) / (charge_rate + discharge_rate + 1e-10)

        return {
            "mean_current": current.mean(),
            "current_variance": current.var(),
            "charge_rate": charge_rate,
            "discharge_rate": discharge_rate,
            "current_asymmetry": current_asymmetry,
        }

    def _extract_efficiency_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract efficiency-related features."""
        efficiency_features = {}

        # Group by cycle
        cycle_stats = data.groupby("cycle_number").agg(
            {"charge_capacity": "mean", "discharge_capacity": "mean"}
        )

        if len(cycle_stats) < 2:
            return {feat: 0.0 for feat in self.EFFICIENCY_FEATURES}

        # Coulombic efficiency
        ce = (cycle_stats["discharge_capacity"] / cycle_stats["charge_capacity"]) * 100
        ce = ce.replace([np.inf, -np.inf], np.nan).dropna()

        if len(ce) > 0:
            efficiency_features["coulombic_efficiency"] = ce.mean()
            efficiency_features["coulombic_efficiency_variance"] = ce.var()
        else:
            efficiency_features["coulombic_efficiency"] = 0
            efficiency_features["coulombic_efficiency_variance"] = 0

        # Energy efficiency (if voltage available)
        if "voltage" in data.columns:
            cycle_energy = data.groupby("cycle_number").agg(
                {"discharge_capacity": "mean", "voltage": "mean"}
            )
            energy_eff = (cycle_energy["discharge_capacity"] * cycle_energy["voltage"]).mean()
            efficiency_features["energy_efficiency"] = energy_eff
        else:
            efficiency_features["energy_efficiency"] = 0

        # Power fade rate
        if "voltage" in data.columns and "current" in data.columns:
            power = (data["voltage"] * abs(data["current"])).groupby(data["cycle_number"]).mean()
            if len(power) > 1:
                x = np.arange(len(power))
                power_fade, _ = np.polyfit(x, power.values, 1)
                efficiency_features["power_fade_rate"] = power_fade
            else:
                efficiency_features["power_fade_rate"] = 0
        else:
            efficiency_features["power_fade_rate"] = 0

        return efficiency_features

    def _extract_statistical_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract statistical features."""
        capacity_by_cycle = data.groupby("cycle_number")["discharge_capacity"].mean()

        if len(capacity_by_cycle) < 3:
            return {feat: 0.0 for feat in self.STATISTICAL_FEATURES}

        # Skewness and kurtosis
        try:
            skewness = stats.skew(capacity_by_cycle.values)
            kurtosis = stats.kurtosis(capacity_by_cycle.values)
        except:
            skewness = 0
            kurtosis = 0

        # Autocorrelation (lag 1)
        if len(capacity_by_cycle) > 1:
            autocorr = capacity_by_cycle.autocorr(lag=1)
            autocorr = autocorr if not np.isnan(autocorr) else 0
        else:
            autocorr = 0

        # Entropy (information content)
        try:
            hist, _ = np.histogram(capacity_by_cycle.values, bins=10)
            hist = hist / hist.sum()  # Normalize
            entropy = stats.entropy(hist + 1e-10)  # Add small value to avoid log(0)
        except:
            entropy = 0

        return {
            "capacity_skewness": skewness,
            "capacity_kurtosis": kurtosis,
            "voltage_autocorr": autocorr,
            "capacity_entropy": entropy,
        }

    def _extract_cycle_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract cycle-related features."""
        cycle_count = data["cycle_number"].nunique()

        # Average cycle time
        if "time_seconds" in data.columns:
            cycle_times = data.groupby("cycle_number")["time_seconds"].max()
            avg_cycle_time = cycle_times.mean()

            # Time at high voltage (> 90% of max voltage)
            if "voltage" in data.columns:
                max_v = data["voltage"].max()
                time_at_high_v = len(data[data["voltage"] > 0.9 * max_v]) / len(data)
            else:
                time_at_high_v = 0

            # Time at high current (> 90% of max current)
            if "current" in data.columns:
                max_i = data["current"].abs().max()
                time_at_high_i = len(data[data["current"].abs() > 0.9 * max_i]) / len(data)
            else:
                time_at_high_i = 0
        else:
            avg_cycle_time = 0
            time_at_high_v = 0
            time_at_high_i = 0

        # Cycles since significant capacity drop (> 2%)
        capacity_by_cycle = data.groupby("cycle_number")["discharge_capacity"].mean()
        initial_cap = capacity_by_cycle.iloc[0]

        significant_drops = capacity_by_cycle < (initial_cap * 0.98)
        if significant_drops.any():
            last_drop_cycle = capacity_by_cycle[significant_drops].index[-1]
            cycles_since_drop = cycle_count - last_drop_cycle
        else:
            cycles_since_drop = cycle_count

        return {
            "cycle_count": cycle_count,
            "avg_cycle_time": avg_cycle_time,
            "cycles_since_capacity_drop": cycles_since_drop,
            "time_at_high_voltage": time_at_high_v,
            "time_at_high_current": time_at_high_i,
        }

    def _extract_temperature_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract temperature-related features."""
        temp = data["temperature"].dropna()

        if len(temp) < 2:
            return {feat: 0.0 for feat in self.TEMPERATURE_FEATURES}

        # Temperature rise rate
        temp_by_cycle = data.groupby("cycle_number")["temperature"].mean()
        if len(temp_by_cycle) > 1:
            x = np.arange(len(temp_by_cycle))
            temp_rise_rate, _ = np.polyfit(x, temp_by_cycle.values, 1)
        else:
            temp_rise_rate = 0

        return {
            "max_temperature": temp.max(),
            "avg_temperature": temp.mean(),
            "temp_variance": temp.var(),
            "temp_rise_rate": temp_rise_rate,
        }

    def _calculate_rul(self, battery_data: pd.DataFrame, current_cycle: int) -> int:
        """
        Calculate Remaining Useful Life (cycles until EOL).

        EOL is defined as when discharge capacity drops below threshold
        of initial capacity.

        Args:
            battery_data: Full battery dataset
            current_cycle: Current cycle number

        Returns:
            RUL in cycles
        """
        # Get initial capacity
        initial_cap = battery_data.groupby("cycle_number")["discharge_capacity"].mean().iloc[0]
        eol_capacity = initial_cap * self.eol_threshold

        # Get future cycles
        future_data = battery_data[battery_data["cycle_number"] > current_cycle]
        future_capacity = future_data.groupby("cycle_number")["discharge_capacity"].mean()

        # Find when capacity drops below EOL threshold
        eol_cycles = future_capacity[future_capacity < eol_capacity]

        if len(eol_cycles) > 0:
            eol_cycle = eol_cycles.index[0]
            rul = eol_cycle - current_cycle
        else:
            # Battery hasn't reached EOL yet
            max_cycle = battery_data["cycle_number"].max()
            rul = max_cycle - current_cycle

        return max(0, rul)  # RUL can't be negative

    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all features."""
        return {
            # Capacity
            "initial_capacity": "Initial discharge capacity (Ah)",
            "current_capacity": "Current discharge capacity (Ah)",
            "capacity_fade_rate": "Linear fade rate (Ah/cycle)",
            "capacity_std": "Standard deviation of capacity",
            "capacity_variance": "Variance of capacity",
            "capacity_range": "Range of capacity values",
            "capacity_trend_strength": "R² of linear trend",
            # Voltage
            "mean_voltage": "Average operating voltage (V)",
            "voltage_drop_rate": "Rate of voltage decline (V/cycle)",
            "voltage_variance": "Variance of voltage",
            "voltage_range": "Voltage range (V)",
            "voltage_peak_count": "Normalized peak count",
            # Current
            "mean_current": "Average current (A)",
            "current_variance": "Variance of current",
            "charge_rate": "Average charging current (A)",
            "discharge_rate": "Average discharging current (A)",
            "current_asymmetry": "Charge/discharge asymmetry",
            # Efficiency
            "coulombic_efficiency": "Average CE (%)",
            "coulombic_efficiency_variance": "Variance of CE",
            "energy_efficiency": "Energy efficiency",
            "power_fade_rate": "Power degradation rate",
            # Statistical
            "capacity_skewness": "Skewness of capacity distribution",
            "capacity_kurtosis": "Kurtosis of capacity distribution",
            "voltage_autocorr": "Voltage autocorrelation (lag 1)",
            "capacity_entropy": "Information entropy",
            # Cycle
            "cycle_count": "Total cycles completed",
            "avg_cycle_time": "Average cycle duration (s)",
            "cycles_since_capacity_drop": "Cycles since last 2% drop",
            "time_at_high_voltage": "Fraction at high voltage",
            "time_at_high_current": "Fraction at high current",
            # Target
            "RUL": "Remaining Useful Life (cycles)",
        }
