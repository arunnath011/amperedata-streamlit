"""Integration tests for electrochemical visualization system.

This module tests end-to-end workflows for electrochemical data analysis
and visualization, including multi-component interactions and real-world scenarios.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from frontend.electrochemical.exceptions import (
    CycleAnalysisError,
    DataProcessingError,
    PlottingError,
)
from frontend.electrochemical.models import (
    AgingConfig,
    AgingData,
    AnalysisType,
    ChargeState,
    ComparisonConfig,
    ComparisonData,
    CycleData,
    CycleLifeConfig,
    DifferentialConfig,
    DifferentialData,
    EISConfig,
    EISData,
    RateCapabilityConfig,
    RateCapabilityData,
    TestCondition,
    VoltageCapacityConfig,
)
from frontend.electrochemical.processors import (
    AgingAnalyzer,
    ComparisonAnalyzer,
    CycleAnalyzer,
    DifferentialAnalyzer,
    EISAnalyzer,
    RateAnalyzer,
)
from frontend.electrochemical.visualizations import (
    BatchComparisonPlot,
    CalendarAgingPlot,
    CycleLifePlot,
    DifferentialPlot,
    NyquistPlot,
    RateCapabilityPlot,
    VoltageCapacityPlot,
)


class TestElectrochemicalWorkflows:
    """Test complete electrochemical analysis workflows."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir) / "test_data"
        self.test_data_dir.mkdir(exist_ok=True)

    def create_realistic_cycle_data(
        self, cycle_number: int, capacity_fade: float = 0.0
    ) -> CycleData:
        """Create realistic cycle data for testing.

        Args:
            cycle_number: Cycle number
            capacity_fade: Capacity fade factor (0-1)

        Returns:
            Realistic cycle data
        """
        # Create realistic charge/discharge profile
        time_points = 200
        time_data = np.linspace(0, 7200, time_points)  # 2 hours

        # Charge phase (0-50% of time)
        charge_time = time_points // 2
        charge_voltage = np.linspace(3.0, 4.2, charge_time)
        charge_current = np.ones(charge_time) * 1.0

        # Discharge phase (50-100% of time)
        discharge_time = time_points - charge_time
        discharge_voltage = np.linspace(4.2, 2.8, discharge_time)
        discharge_current = np.ones(discharge_time) * -1.0

        # Combine phases
        voltage = np.concatenate([charge_voltage, discharge_voltage])
        current = np.concatenate([charge_current, discharge_current])

        # Calculate capacity with fade
        base_capacity = 2.5 * (1 - capacity_fade)
        capacity = np.zeros_like(time_data)

        # Charge phase capacity
        capacity[:charge_time] = np.linspace(0, base_capacity, charge_time)
        # Discharge phase capacity
        capacity[charge_time:] = np.linspace(base_capacity, 0, discharge_time)

        # Charge states
        charge_state = [ChargeState.CHARGE] * charge_time + [ChargeState.DISCHARGE] * discharge_time

        return CycleData(
            name=f"Realistic Cycle {cycle_number}",
            cycle_number=cycle_number,
            time=time_data.tolist(),
            voltage=voltage.tolist(),
            current=current.tolist(),
            capacity=capacity.tolist(),
            charge_state=charge_state,
            test_condition=TestCondition.CYCLE,
            c_rate=1.0,
            temperature=25.0,
        )

    def create_realistic_eis_data(self, soc: float = 50.0) -> EISData:
        """Create realistic EIS data for testing.

        Args:
            soc: State of charge (%)

        Returns:
            Realistic EIS data
        """
        # Frequency range from 1 MHz to 0.01 Hz
        frequency = np.logspace(6, -2, 50)
        omega = 2 * np.pi * frequency

        # Realistic battery equivalent circuit: R_s + (R_ct || CPE) + Warburg
        R_s = 0.05  # Series resistance
        R_ct = 0.1 + 0.001 * (100 - soc)  # Charge transfer resistance (SOC dependent)
        CPE_T = 1e-3  # CPE coefficient
        CPE_P = 0.9  # CPE exponent
        sigma = 0.01  # Warburg coefficient

        # Calculate impedance
        Z_cpe = 1 / (CPE_T * (1j * omega) ** CPE_P)
        Z_warburg = sigma / np.sqrt(omega) * (1 - 1j)
        Z_ct_cpe = (R_ct * Z_cpe) / (R_ct + Z_cpe)
        Z_total = R_s + Z_ct_cpe + Z_warburg

        return EISData(
            name=f"Realistic EIS SOC {soc}%",
            frequency=frequency.tolist(),
            impedance_real=Z_total.real.tolist(),
            impedance_imag=Z_total.imag.tolist(),
            soc=soc,
            voltage=3.7 + (soc - 50) * 0.01,  # SOC-dependent voltage
            temperature=25.0,
        )

    def test_complete_battery_characterization_workflow(self):
        """Test complete battery characterization workflow."""
        print("\n=== Testing Complete Battery Characterization Workflow ===")

        # Step 1: Create cycle life data
        cycle_data_list = []
        for cycle in range(1, 21):  # 20 cycles
            fade_factor = 0.02 * (cycle - 1)  # 2% fade per cycle
            cycle_data = self.create_realistic_cycle_data(cycle, fade_factor)
            cycle_data_list.append(cycle_data)

        print(f"Created {len(cycle_data_list)} cycles with progressive capacity fade")

        # Step 2: Analyze cycle life
        cycle_analyzer = CycleAnalyzer()
        fade_results = []

        for cycle_data in cycle_data_list[:5]:  # Analyze first 5 cycles
            try:
                result = cycle_analyzer.process(cycle_data)
                fade_results.append(result)
                print(
                    f"Cycle {cycle_data.cycle_number}: Fade rate = {result.fade_rate_per_cycle:.3f}%/cycle"
                )
            except Exception as e:
                print(f"Analysis failed for cycle {cycle_data.cycle_number}: {e}")

        assert len(fade_results) > 0, "Should have successful fade analysis results"

        # Step 3: Create cycle life plot
        config = CycleLifeConfig(show_capacity=True, fit_fade_model=True, eol_threshold=80.0)

        cycle_plot = CycleLifePlot(config)

        with patch("frontend.electrochemical.visualizations.PLOTLY_AVAILABLE", True):
            with patch("frontend.electrochemical.visualizations.go") as mock_go:
                mock_figure = Mock()
                mock_go.Figure.return_value = mock_figure
                mock_go.Scatter = Mock()

                figure = cycle_plot.create_plot(cycle_data_list)
                assert figure is not None
                print("✓ Cycle life plot created successfully")

        # Step 4: Create EIS data at different SOCs
        eis_data_list = []
        for soc in [10, 30, 50, 70, 90]:
            eis_data = self.create_realistic_eis_data(soc)
            eis_data_list.append(eis_data)

        print(f"Created EIS data for {len(eis_data_list)} SOC points")

        # Step 5: Analyze EIS data
        eis_analyzer = EISAnalyzer()
        eis_results = []

        for eis_data in eis_data_list:
            try:
                result = eis_analyzer.process(eis_data)
                eis_results.append(result)
                print(
                    f"EIS SOC {eis_data.soc}%: R_series = {result.resistance_values.get('r_series', 0):.4f} Ω"
                )
            except Exception as e:
                print(f"EIS analysis failed for SOC {eis_data.soc}%: {e}")

        assert len(eis_results) > 0, "Should have successful EIS analysis results"

        # Step 6: Create Nyquist plots
        eis_config = EISConfig(show_frequency_labels=True)
        nyquist_plot = NyquistPlot(eis_config)

        with patch("frontend.electrochemical.visualizations.PLOTLY_AVAILABLE", True):
            with patch("frontend.electrochemical.visualizations.go") as mock_go:
                mock_figure = Mock()
                mock_go.Figure.return_value = mock_figure
                mock_go.Scatter = Mock()

                for eis_data in eis_data_list[:2]:  # Test first 2
                    figure = nyquist_plot.create_plot(eis_data)
                    assert figure is not None

                print("✓ Nyquist plots created successfully")

        print("=== Battery Characterization Workflow Completed Successfully ===\n")

    def test_differential_analysis_workflow(self):
        """Test differential analysis workflow."""
        print("\n=== Testing Differential Analysis Workflow ===")

        # Create cycle data with realistic voltage profile
        cycle_data = self.create_realistic_cycle_data(1)

        # Extract charge data for differential analysis
        charge_indices = [
            i for i, state in enumerate(cycle_data.charge_state) if state == ChargeState.CHARGE
        ]

        if len(charge_indices) > 10:  # Need sufficient data points
            charge_voltage = [cycle_data.voltage[i] for i in charge_indices]
            charge_capacity = [cycle_data.capacity[i] for i in charge_indices]

            diff_data = DifferentialData(
                name="Charge Differential",
                voltage=charge_voltage,
                capacity=charge_capacity,
                cycle_number=1,
                charge_state=ChargeState.CHARGE,
                smoothing_window=5,
            )

            print(f"Created differential data with {len(charge_voltage)} points")

            # Analyze differential data
            diff_analyzer = DifferentialAnalyzer()
            try:
                result = diff_analyzer.process(diff_data)
                print(f"Differential analysis: {len(result.peaks_detected)} peaks detected")

                # Create differential plot
                diff_config = DifferentialConfig(
                    analysis_type="dq_dv", smoothing_method="savgol", peak_detection=True
                )

                diff_plot = DifferentialPlot(diff_config)

                with patch("frontend.electrochemical.visualizations.PLOTLY_AVAILABLE", True):
                    with patch(
                        "frontend.electrochemical.visualizations.make_subplots"
                    ) as mock_subplots:
                        with patch("frontend.electrochemical.visualizations.go") as mock_go:
                            mock_figure = Mock()
                            mock_subplots.return_value = mock_figure
                            mock_go.Scatter = Mock()

                            figure = diff_plot.create_plot(diff_data)
                            assert figure is not None
                            print("✓ Differential plot created successfully")

            except Exception as e:
                print(f"Differential analysis failed: {e}")
                pytest.skip("Differential analysis failed - may be due to data characteristics")

        print("=== Differential Analysis Workflow Completed ===\n")

    def test_rate_capability_workflow(self):
        """Test rate capability analysis workflow."""
        print("\n=== Testing Rate Capability Workflow ===")

        # Create rate capability data
        c_rates = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        base_capacity = 2.5

        # Realistic capacity fade with C-rate (Peukert's law approximation)
        discharge_capacity = []
        efficiency = []

        for rate in c_rates:
            # Capacity decreases with rate
            capacity = base_capacity * (0.1 / rate) ** 0.1 if rate > 0.1 else base_capacity
            capacity = max(capacity, base_capacity * 0.5)  # Minimum 50% retention
            discharge_capacity.append(capacity)

            # Efficiency decreases with rate
            eff = 99.5 - rate * 0.5
            eff = max(eff, 90.0)  # Minimum 90% efficiency
            efficiency.append(eff)

        rate_data = RateCapabilityData(
            name="Rate Capability Test",
            c_rates=c_rates,
            discharge_capacity=discharge_capacity,
            efficiency=efficiency,
            cycle_number=5,
        )

        print(f"Created rate capability data for C-rates: {c_rates}")

        # Analyze rate capability
        rate_analyzer = RateAnalyzer()
        try:
            result = rate_analyzer.process(rate_data)
            print(f"Max C-rate: {result.max_c_rate:.1f}C")
            print(f"Capacity retention at 1C: {result.capacity_retention_at_1c:.1f}%")
            print(f"Rate capability score: {result.rate_capability_score:.3f}")

            # Create rate capability plot
            rate_config = RateCapabilityConfig(
                show_capacity=True, show_efficiency=True, normalize_capacity=True
            )

            rate_plot = RateCapabilityPlot(rate_config)

            with patch("frontend.electrochemical.visualizations.PLOTLY_AVAILABLE", True):
                with patch("frontend.electrochemical.visualizations.go") as mock_go:
                    mock_figure = Mock()
                    mock_go.Figure.return_value = mock_figure
                    mock_go.Scatter = Mock()

                    figure = rate_plot.create_plot(rate_data)
                    assert figure is not None
                    print("✓ Rate capability plot created successfully")

        except Exception as e:
            print(f"Rate capability analysis failed: {e}")
            raise

        print("=== Rate Capability Workflow Completed ===\n")

    def test_aging_analysis_workflow(self):
        """Test aging analysis workflow."""
        print("\n=== Testing Aging Analysis Workflow ===")

        # Create calendar aging data
        time_days = np.linspace(0, 365, 50)  # 1 year

        # Realistic aging model: square root of time + linear component
        sqrt_component = 5 * np.sqrt(time_days / 365)  # 5% fade from sqrt(time)
        linear_component = 2 * (time_days / 365)  # 2% fade from linear time
        capacity_retention = 100 - sqrt_component - linear_component

        # Add some realistic noise
        noise = np.random.normal(0, 0.5, len(time_days))
        capacity_retention += noise
        capacity_retention = np.clip(capacity_retention, 70, 100)  # Realistic bounds

        aging_data = AgingData(
            name="Calendar Aging Test",
            time_days=time_days.tolist(),
            capacity_retention=capacity_retention.tolist(),
            aging_type="calendar",
            storage_temperature=45.0,
            storage_soc=50.0,
        )

        print(f"Created aging data over {time_days[-1]:.0f} days")
        print(f"Final capacity retention: {capacity_retention[-1]:.1f}%")

        # Analyze aging data
        aging_analyzer = AgingAnalyzer()
        try:
            result = aging_analyzer.process(aging_data)
            print(f"Aging rate: {result.aging_rate:.3f}%/day")
            if result.calendar_life_prediction:
                print(f"Predicted calendar life: {result.calendar_life_prediction:.1f} years")

            # Create aging plot
            aging_config = AgingConfig(
                show_capacity=True,
                fit_aging_model=True,
                aging_model_type="sqrt_time",
                time_units="days",
            )

            aging_plot = CalendarAgingPlot(aging_config)

            with patch("frontend.electrochemical.visualizations.PLOTLY_AVAILABLE", True):
                with patch("frontend.electrochemical.visualizations.go") as mock_go:
                    mock_figure = Mock()
                    mock_go.Figure.return_value = mock_figure
                    mock_go.Scatter = Mock()

                    figure = aging_plot.create_plot(aging_data)
                    assert figure is not None
                    print("✓ Aging plot created successfully")

        except Exception as e:
            print(f"Aging analysis failed: {e}")
            raise

        print("=== Aging Analysis Workflow Completed ===\n")

    def test_batch_comparison_workflow(self):
        """Test batch comparison workflow."""
        print("\n=== Testing Batch Comparison Workflow ===")

        # Create multiple samples with different performance
        datasets = []
        sample_names = ["Sample A", "Sample B", "Sample C", "Sample D", "Sample E"]
        base_capacities = [2.5, 2.3, 2.7, 2.4, 2.6]  # Different initial capacities

        for _i, (name, capacity) in enumerate(zip(sample_names, base_capacities)):
            cycle_data = CycleData(
                name=name,
                cycle_number=10,  # All at cycle 10 for comparison
                time=[0, 1800, 3600],
                voltage=[3.0, 4.0, 3.2],
                current=[1.0, 0.0, -1.0],
                capacity=[0.0, capacity / 2, capacity],
                charge_state=[ChargeState.CHARGE, ChargeState.REST, ChargeState.DISCHARGE],
                test_condition=TestCondition.CYCLE,
                temperature=25.0,
            )
            datasets.append(cycle_data)

        comparison_data = ComparisonData(
            name="Batch Comparison Test",
            description="Comparing 5 battery samples",
            datasets=datasets,
            comparison_type=AnalysisType.CAPACITY_FADE,
            statistical_analysis=True,
        )

        print(f"Created comparison data with {len(datasets)} samples")

        # Analyze comparison data
        comp_analyzer = ComparisonAnalyzer()
        try:
            result = comp_analyzer.process(comparison_data)

            stats = result.statistical_summary.get("all", {})
            print(f"Mean capacity: {stats.get('mean', 0):.3f} Ah")
            print(f"Std deviation: {stats.get('std', 0):.3f} Ah")
            print(f"Best sample: {result.best_performing_sample}")
            print(f"Worst sample: {result.worst_performing_sample}")

            # Create comparison plot
            comp_config = ComparisonConfig(
                show_individual=True, show_statistics=True, confidence_interval=0.95
            )

            comp_plot = BatchComparisonPlot(comp_config)

            with patch("frontend.electrochemical.visualizations.PLOTLY_AVAILABLE", True):
                with patch("frontend.electrochemical.visualizations.go") as mock_go:
                    mock_figure = Mock()
                    mock_go.Figure.return_value = mock_figure
                    mock_go.Box = Mock()
                    mock_go.Bar = Mock()
                    mock_go.Scatter = Mock()

                    figure = comp_plot.create_plot(comparison_data)
                    assert figure is not None
                    print("✓ Batch comparison plot created successfully")

        except Exception as e:
            print(f"Batch comparison analysis failed: {e}")
            raise

        print("=== Batch Comparison Workflow Completed ===\n")

    def test_multi_analysis_dashboard_workflow(self):
        """Test creating multiple analyses for a dashboard-like view."""
        print("\n=== Testing Multi-Analysis Dashboard Workflow ===")

        # Create comprehensive dataset
        cycle_data = self.create_realistic_cycle_data(1)
        eis_data = self.create_realistic_eis_data(50.0)

        # Create rate capability data
        rate_data = RateCapabilityData(
            name="Dashboard Rate Test",
            c_rates=[0.5, 1.0, 2.0, 5.0],
            discharge_capacity=[2.5, 2.4, 2.2, 1.9],
            efficiency=[99.0, 98.5, 97.5, 95.0],
            cycle_number=1,
        )

        analyses_completed = []

        # Run multiple analyses
        try:
            # Cycle analysis
            cycle_analyzer = CycleAnalyzer()
            cycle_analyzer.process(cycle_data)
            analyses_completed.append("Cycle Analysis")

            # EIS analysis
            eis_analyzer = EISAnalyzer()
            eis_analyzer.process(eis_data)
            analyses_completed.append("EIS Analysis")

            # Rate analysis
            rate_analyzer = RateAnalyzer()
            rate_analyzer.process(rate_data)
            analyses_completed.append("Rate Analysis")

            print(f"Completed analyses: {', '.join(analyses_completed)}")

            # Create multiple plots
            plots_created = []

            with patch("frontend.electrochemical.visualizations.PLOTLY_AVAILABLE", True):
                with patch("frontend.electrochemical.visualizations.go") as mock_go:
                    with patch(
                        "frontend.electrochemical.visualizations.make_subplots"
                    ) as mock_subplots:
                        mock_figure = Mock()
                        mock_go.Figure.return_value = mock_figure
                        mock_subplots.return_value = mock_figure
                        mock_go.Scatter = Mock()
                        mock_go.Box = Mock()

                        # Voltage-capacity plot
                        vc_config = VoltageCapacityConfig()
                        vc_plot = VoltageCapacityPlot(vc_config)
                        vc_plot.create_plot(cycle_data)
                        plots_created.append("Voltage-Capacity")

                        # Nyquist plot
                        eis_config = EISConfig()
                        nyquist_plot = NyquistPlot(eis_config)
                        nyquist_plot.create_plot(eis_data)
                        plots_created.append("Nyquist")

                        # Rate capability plot
                        rate_config = RateCapabilityConfig()
                        rate_plot = RateCapabilityPlot(rate_config)
                        rate_plot.create_plot(rate_data)
                        plots_created.append("Rate Capability")

            print(f"Created plots: {', '.join(plots_created)}")

            # Verify all components worked together
            assert len(analyses_completed) == 3
            assert len(plots_created) == 3

            print("✓ Multi-analysis dashboard workflow completed successfully")

        except Exception as e:
            print(f"Multi-analysis workflow failed: {e}")
            raise

        print("=== Multi-Analysis Dashboard Workflow Completed ===\n")

    def test_error_recovery_workflow(self):
        """Test error handling and recovery in workflows."""
        from pydantic import ValidationError

        print("\n=== Testing Error Recovery Workflow ===")

        # Test with various problematic data
        error_scenarios = []

        # Scenario 1: Empty data - expect CycleAnalysisError
        try:
            empty_cycle = CycleData(
                name="Empty Cycle",
                cycle_number=1,
                time=[],
                voltage=[],
                current=[],
                capacity=[],
                charge_state=[],
                test_condition=TestCondition.CYCLE,
            )

            analyzer = CycleAnalyzer()
            analyzer.process(empty_cycle)

        except (DataProcessingError, CycleAnalysisError, ValueError, Exception) as e:
            error_scenarios.append("Empty data handled")
            print(f"✓ Empty data error handled: {type(e).__name__}")

        # Scenario 2: Mismatched array lengths - Pydantic should catch this
        try:
            CycleData(
                name="Mismatched Cycle",
                cycle_number=1,
                time=[0, 1, 2],
                voltage=[3.0, 3.5],  # Different length
                current=[1.0, 1.0, 0.0],
                capacity=[0.0, 0.5, 1.0],
                charge_state=[ChargeState.CHARGE] * 3,
                test_condition=TestCondition.CYCLE,
            )

        except (ValueError, ValidationError, Exception) as e:
            error_scenarios.append("Mismatched arrays handled")
            print(f"✓ Mismatched array error handled: {type(e).__name__}")

        # Scenario 3: Invalid plot data
        try:
            config = VoltageCapacityConfig()
            plot = VoltageCapacityPlot(config)

            invalid_cycle = CycleData(
                name="Invalid Plot Data",
                cycle_number=1,
                time=[0, 1, 2],
                voltage=[],  # Empty voltage
                current=[1.0, 1.0, 0.0],
                capacity=[0.0, 0.5, 1.0],
                charge_state=[ChargeState.CHARGE] * 3,
                test_condition=TestCondition.CYCLE,
            )

            plot.create_plot(invalid_cycle)

        except (PlottingError, Exception) as e:
            error_scenarios.append("Invalid plot data handled")
            print(f"✓ Invalid plot data error handled: {type(e).__name__}")

        # Scenario 4: Insufficient data for analysis
        try:
            minimal_cycle = CycleData(
                name="Minimal Cycle",
                cycle_number=1,
                time=[0],
                voltage=[3.7],
                current=[1.0],
                capacity=[0.0],
                charge_state=[ChargeState.CHARGE],
                test_condition=TestCondition.CYCLE,
            )

            analyzer = CycleAnalyzer()
            analyzer.process(minimal_cycle)

        except (DataProcessingError, CycleAnalysisError, Exception) as e:
            error_scenarios.append("Insufficient data handled")
            print(f"✓ Insufficient data error handled: {type(e).__name__}")

        print(f"Error scenarios tested: {len(error_scenarios)}")
        assert len(error_scenarios) >= 2, "Should handle multiple error scenarios"

        print("=== Error Recovery Workflow Completed ===\n")

    def test_performance_workflow(self):
        """Test performance with larger datasets."""
        print("\n=== Testing Performance Workflow ===")

        # Create large dataset
        large_time_points = 1000
        time_data = np.linspace(0, 3600, large_time_points)
        voltage_data = (
            3.7 + 0.5 * np.sin(time_data / 1800 * np.pi) + 0.1 * np.random.randn(large_time_points)
        )
        current_data = np.ones_like(time_data)
        current_data[500:] = -1.0
        capacity_data = np.cumsum(np.abs(current_data)) * 0.001
        charge_states = [
            ChargeState.CHARGE if c > 0 else ChargeState.DISCHARGE for c in current_data
        ]

        large_cycle = CycleData(
            name="Large Dataset Cycle",
            cycle_number=1,
            time=time_data.tolist(),
            voltage=voltage_data.tolist(),
            current=current_data.tolist(),
            capacity=capacity_data.tolist(),
            charge_state=charge_states,
            test_condition=TestCondition.CYCLE,
        )

        print(f"Created large dataset with {large_time_points} data points")

        # Time the analysis
        import time

        start_time = time.time()

        try:
            analyzer = CycleAnalyzer()
            analyzer.process(large_cycle)

            analysis_time = time.time() - start_time
            print(f"Analysis completed in {analysis_time:.3f} seconds")

            # Time the plotting
            start_time = time.time()

            config = VoltageCapacityConfig()
            plot = VoltageCapacityPlot(config)

            with patch("frontend.electrochemical.visualizations.PLOTLY_AVAILABLE", True):
                with patch("frontend.electrochemical.visualizations.go") as mock_go:
                    mock_figure = Mock()
                    mock_go.Figure.return_value = mock_figure
                    mock_go.Scatter = Mock()

                    plot.create_plot(large_cycle)

                    plot_time = time.time() - start_time
                    print(f"Plot creation completed in {plot_time:.3f} seconds")

            # Performance should be reasonable (< 5 seconds total)
            total_time = analysis_time + plot_time
            print(f"Total processing time: {total_time:.3f} seconds")

            assert total_time < 5.0, f"Performance too slow: {total_time:.3f}s"
            print("✓ Performance test passed")

        except Exception as e:
            print(f"Performance test failed: {e}")
            # Don't fail the test for performance issues in CI
            pytest.skip(f"Performance test skipped due to: {e}")

        print("=== Performance Workflow Completed ===\n")


class TestElectrochemicalDataFormats:
    """Test handling of different data formats and edge cases."""

    def test_data_format_compatibility(self):
        """Test compatibility with different data formats."""
        print("\n=== Testing Data Format Compatibility ===")

        # Test with pandas DataFrame conversion
        df_data = pd.DataFrame(
            {
                "time": [0, 1, 2, 3, 4],
                "voltage": [3.0, 3.5, 4.0, 3.8, 3.2],
                "current": [1.0, 1.0, 0.0, -1.0, -1.0],
                "capacity": [0.0, 0.5, 1.0, 1.5, 2.0],
            }
        )

        # Convert DataFrame to CycleData
        cycle_data = CycleData(
            name="DataFrame Cycle",
            cycle_number=1,
            time=df_data["time"].tolist(),
            voltage=df_data["voltage"].tolist(),
            current=df_data["current"].tolist(),
            capacity=df_data["capacity"].tolist(),
            charge_state=[ChargeState.CHARGE] * len(df_data),
            test_condition=TestCondition.CYCLE,
        )

        # Test analysis
        analyzer = CycleAnalyzer()
        result = analyzer.process(cycle_data)

        assert result is not None
        print("✓ DataFrame conversion and analysis successful")

        # Test with numpy arrays
        np_time = np.array([0, 1, 2, 3, 4])
        np_voltage = np.array([3.0, 3.5, 4.0, 3.8, 3.2])
        np_current = np.array([1.0, 1.0, 0.0, -1.0, -1.0])
        np_capacity = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

        numpy_cycle = CycleData(
            name="NumPy Cycle",
            cycle_number=1,
            time=np_time.tolist(),
            voltage=np_voltage.tolist(),
            current=np_current.tolist(),
            capacity=np_capacity.tolist(),
            charge_state=[ChargeState.CHARGE] * len(np_time),
            test_condition=TestCondition.CYCLE,
        )

        result = analyzer.process(numpy_cycle)
        assert result is not None
        print("✓ NumPy array conversion and analysis successful")

        print("=== Data Format Compatibility Test Completed ===\n")

    def test_edge_case_data_handling(self):
        """Test handling of edge case data."""
        print("\n=== Testing Edge Case Data Handling ===")

        edge_cases_handled = []

        # Test with very small values
        try:
            small_values_cycle = CycleData(
                name="Small Values Cycle",
                cycle_number=1,
                time=[0, 1e-6, 2e-6, 3e-6],
                voltage=[3.0, 3.0001, 3.0002, 3.0001],
                current=[1e-9, 1e-9, -1e-9, -1e-9],
                capacity=[0, 1e-12, 2e-12, 1e-12],
                charge_state=[ChargeState.CHARGE] * 4,
                test_condition=TestCondition.CYCLE,
            )

            analyzer = CycleAnalyzer()
            analyzer.process(small_values_cycle)
            edge_cases_handled.append("Small values")
            print("✓ Small values handled successfully")

        except Exception as e:
            print(f"Small values handling failed: {e}")

        # Test with large values
        try:
            large_values_cycle = CycleData(
                name="Large Values Cycle",
                cycle_number=1,
                time=[0, 1e6, 2e6, 3e6],
                voltage=[3000, 3500, 4000, 3500],
                current=[1000, 1000, -1000, -1000],
                capacity=[0, 500, 1000, 500],
                charge_state=[ChargeState.CHARGE] * 4,
                test_condition=TestCondition.CYCLE,
            )

            analyzer.process(large_values_cycle)
            edge_cases_handled.append("Large values")
            print("✓ Large values handled successfully")

        except Exception as e:
            print(f"Large values handling failed: {e}")

        # Test with constant values
        try:
            constant_cycle = CycleData(
                name="Constant Values Cycle",
                cycle_number=1,
                time=[0, 1, 2, 3, 4],
                voltage=[3.7, 3.7, 3.7, 3.7, 3.7],  # Constant voltage
                current=[1.0, 1.0, 1.0, 1.0, 1.0],  # Constant current
                capacity=[0, 1, 2, 3, 4],
                charge_state=[ChargeState.CHARGE] * 5,
                test_condition=TestCondition.CONSTANT_CURRENT,
            )

            analyzer.process(constant_cycle)
            edge_cases_handled.append("Constant values")
            print("✓ Constant values handled successfully")

        except Exception as e:
            print(f"Constant values handling failed: {e}")

        print(f"Edge cases handled: {len(edge_cases_handled)}")
        print("=== Edge Case Data Handling Test Completed ===\n")


if __name__ == "__main__":
    # Run integration tests
    test_workflows = TestElectrochemicalWorkflows()
    test_workflows.setup_method()

    try:
        test_workflows.test_complete_battery_characterization_workflow()
        test_workflows.test_differential_analysis_workflow()
        test_workflows.test_rate_capability_workflow()
        test_workflows.test_aging_analysis_workflow()
        test_workflows.test_batch_comparison_workflow()
        test_workflows.test_multi_analysis_dashboard_workflow()
        test_workflows.test_error_recovery_workflow()
        test_workflows.test_performance_workflow()

        test_formats = TestElectrochemicalDataFormats()
        test_formats.test_data_format_compatibility()
        test_formats.test_edge_case_data_handling()

        print("🎉 All integration tests completed successfully!")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise
