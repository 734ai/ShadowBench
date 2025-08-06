"""
Energy Monitoring Integration for ShadowBench Benchmark Runner
Provides seamless integration of energy monitoring into benchmarking workflows.
"""

import logging
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from .energy_monitor import EnergyMonitor, EnergyMeasurement
import time
import json


class BenchmarkEnergyTracker:
    """
    Energy tracking wrapper for benchmark operations.
    Provides context management and automatic energy profiling.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.energy_monitor = EnergyMonitor(config)
        
        # Tracking data
        self.benchmark_sessions: List[Dict[str, Any]] = []
        self.current_session: Optional[Dict[str, Any]] = None
    
    @contextmanager
    def track_benchmark(self, benchmark_name: str, metadata: Optional[Dict] = None):
        """
        Context manager for tracking energy consumption during benchmark execution.
        
        Args:
            benchmark_name: Name of the benchmark being executed
            metadata: Additional metadata for the benchmark session
        """
        session_metadata = metadata or {}
        session_start = time.time()
        
        # Start energy monitoring
        if not self.energy_monitor.start_monitoring():
            self.logger.error("Failed to start energy monitoring")
            yield None
            return
        
        # Initialize session tracking
        self.current_session = {
            'benchmark_name': benchmark_name,
            'start_time': session_start,
            'metadata': session_metadata,
            'energy_data': [],
            'milestones': []
        }
        
        try:
            self.logger.info(f"Started energy tracking for benchmark: {benchmark_name}")
            yield self
            
        except Exception as e:
            self.logger.error(f"Benchmark execution failed: {e}")
            self.current_session['error'] = str(e)
            raise
            
        finally:
            # Stop monitoring and collect final data
            session_summary = self.energy_monitor.stop_monitoring()
            session_end = time.time()
            
            self.current_session.update({
                'end_time': session_end,
                'duration_seconds': session_end - session_start,
                'session_summary': session_summary,
                'carbon_footprint': self.energy_monitor.calculate_carbon_footprint()
            })
            
            # Store completed session
            self.benchmark_sessions.append(self.current_session.copy())
            
            self.logger.info(f"Completed energy tracking for benchmark: {benchmark_name}")
            self.current_session = None
    
    def add_milestone(self, milestone_name: str, metadata: Optional[Dict] = None):
        """Add a milestone marker during benchmark execution."""
        if not self.current_session:
            self.logger.warning("No active benchmark session for milestone")
            return
        
        current_measurement = self.energy_monitor.get_current_measurement()
        milestone = {
            'name': milestone_name,
            'timestamp': time.time(),
            'metadata': metadata or {},
            'energy_measurement': current_measurement
        }
        
        self.current_session['milestones'].append(milestone)
        self.logger.debug(f"Added milestone: {milestone_name}")
    
    def get_real_time_metrics(self) -> Optional[Dict[str, Any]]:
        """Get current real-time energy metrics."""
        if not self.energy_monitor.is_monitoring:
            return None
        
        current_measurement = self.energy_monitor.get_current_measurement()
        if not current_measurement:
            return None
        
        # Get recent efficiency recommendations
        recommendations = self.energy_monitor.get_efficiency_recommendations()
        
        return {
            'current_measurement': {
                'timestamp': current_measurement.timestamp,
                'cpu_usage_percent': current_measurement.cpu_usage_percent,
                'gpu_usage_percent': current_measurement.gpu_usage_percent,
                'memory_usage_mb': current_measurement.memory_usage_mb,
                'system_power_watts': current_measurement.system_power_watts,
                'temperature_celsius': current_measurement.temperature_celsius
            },
            'session_info': {
                'benchmark_name': self.current_session['benchmark_name'] if self.current_session else None,
                'session_duration': time.time() - self.current_session['start_time'] if self.current_session else 0,
                'milestones_count': len(self.current_session['milestones']) if self.current_session else 0
            },
            'efficiency_recommendations': recommendations
        }
    
    def generate_energy_report(self, benchmark_name: Optional[str] = None) -> str:
        """Generate comprehensive energy consumption report."""
        if benchmark_name:
            sessions = [s for s in self.benchmark_sessions if s['benchmark_name'] == benchmark_name]
        else:
            sessions = self.benchmark_sessions
        
        if not sessions:
            return "No benchmark sessions found for energy report."
        
        # Calculate aggregate metrics
        total_energy = sum(s.get('carbon_footprint', {}).get('total_energy_kwh', 0) for s in sessions)
        total_carbon = sum(s.get('carbon_footprint', {}).get('carbon_emission_kg', 0) for s in sessions)
        total_duration = sum(s.get('duration_seconds', 0) for s in sessions)
        
        # Generate report
        report_lines = [
            "ShadowBench Energy Consumption Report",
            "=" * 50,
            f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Benchmark Sessions: {len(sessions)}",
            f"Total Duration: {total_duration/3600:.2f} hours",
            "",
            "Energy Consumption Summary:",
            f"  Total Energy: {total_energy:.4f} kWh",
            f"  Carbon Emissions: {total_carbon:.6f} kg CO2",
            f"  Efficiency: {(total_energy/total_duration*3600):.4f} kWh/hour",
            ""
        ]
        
        # Add per-benchmark breakdown
        if len(sessions) > 1:
            report_lines.append("Per-Benchmark Breakdown:")
            report_lines.append("-" * 30)
            
            for session in sessions:
                carbon_footprint = session.get('carbon_footprint', {})
                energy_kwh = carbon_footprint.get('total_energy_kwh', 0)
                carbon_kg = carbon_footprint.get('carbon_emission_kg', 0)
                duration_min = session.get('duration_seconds', 0) / 60
                
                report_lines.extend([
                    f"Benchmark: {session['benchmark_name']}",
                    f"  Duration: {duration_min:.1f} minutes",
                    f"  Energy: {energy_kwh:.4f} kWh",
                    f"  Carbon: {carbon_kg:.6f} kg CO2",
                    f"  Milestones: {len(session.get('milestones', []))}",
                    ""
                ])
        
        # Add environmental impact context
        if total_carbon > 0:
            equivalent_metrics = sessions[0].get('carbon_footprint', {}).get('equivalent_metrics', {})
            if equivalent_metrics:
                report_lines.extend([
                    "Environmental Impact Context:",
                    f"  Equivalent to {equivalent_metrics.get('car_miles_equivalent', 0):.1f} miles driven",
                    f"  Requires {equivalent_metrics.get('tree_months_to_offset', 0):.1f} tree-months to offset",
                    f"  Equal to {equivalent_metrics.get('smartphone_charges', 0):.0f} smartphone charges",
                    ""
                ])
        
        # Add efficiency recommendations
        recent_session = sessions[-1]  # Most recent session
        if 'session_summary' in recent_session:
            recommendations = recent_session['session_summary'].get('efficiency_recommendations', [])
            if recommendations:
                report_lines.extend([
                    "Efficiency Recommendations:",
                    "-" * 30
                ])
                for rec in recommendations:
                    priority_marker = "❗" if rec['priority'] == 'high' else "⚠️" if rec['priority'] == 'medium' else "ℹ️"
                    report_lines.append(f"{priority_marker} {rec['recommendation']}")
                    if 'metric' in rec:
                        report_lines.append(f"   Metric: {rec['metric']}")
                report_lines.append("")
        
        return "\n".join(report_lines)
    
    def export_detailed_data(self, filepath: str, format: str = 'json'):
        """Export detailed energy monitoring data."""
        export_data = {
            'export_timestamp': time.time(),
            'export_format': format,
            'benchmark_sessions': []
        }
        
        for session in self.benchmark_sessions:
            session_data = {
                'benchmark_name': session['benchmark_name'],
                'start_time': session['start_time'],
                'end_time': session['end_time'],
                'duration_seconds': session['duration_seconds'],
                'metadata': session['metadata'],
                'milestones': session['milestones'],
                'carbon_footprint': session.get('carbon_footprint', {}),
                'session_summary': session.get('session_summary', {})
            }
            export_data['benchmark_sessions'].append(session_data)
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Detailed energy data exported to {filepath}")
    
    def compare_benchmarks(self, benchmark_names: List[str]) -> Dict[str, Any]:
        """Compare energy efficiency between different benchmarks."""
        comparison_data = {}
        
        for benchmark_name in benchmark_names:
            sessions = [s for s in self.benchmark_sessions if s['benchmark_name'] == benchmark_name]
            
            if not sessions:
                comparison_data[benchmark_name] = {'error': 'No sessions found'}
                continue
            
            # Calculate average metrics
            avg_energy = sum(s.get('carbon_footprint', {}).get('total_energy_kwh', 0) for s in sessions) / len(sessions)
            avg_carbon = sum(s.get('carbon_footprint', {}).get('carbon_emission_kg', 0) for s in sessions) / len(sessions)
            avg_duration = sum(s.get('duration_seconds', 0) for s in sessions) / len(sessions)
            
            comparison_data[benchmark_name] = {
                'session_count': len(sessions),
                'average_energy_kwh': avg_energy,
                'average_carbon_kg': avg_carbon,
                'average_duration_seconds': avg_duration,
                'energy_efficiency': avg_energy / (avg_duration / 3600) if avg_duration > 0 else 0,  # kWh per hour
                'carbon_efficiency': avg_carbon / (avg_duration / 3600) if avg_duration > 0 else 0   # kg CO2 per hour
            }
        
        # Add ranking
        valid_benchmarks = [(name, data) for name, data in comparison_data.items() if 'error' not in data]
        
        if len(valid_benchmarks) > 1:
            # Rank by energy efficiency (lower is better)
            energy_ranking = sorted(valid_benchmarks, key=lambda x: x[1]['energy_efficiency'])
            carbon_ranking = sorted(valid_benchmarks, key=lambda x: x[1]['carbon_efficiency'])
            
            comparison_data['rankings'] = {
                'most_energy_efficient': energy_ranking[0][0] if energy_ranking else None,
                'least_energy_efficient': energy_ranking[-1][0] if energy_ranking else None,
                'lowest_carbon_footprint': carbon_ranking[0][0] if carbon_ranking else None,
                'highest_carbon_footprint': carbon_ranking[-1][0] if carbon_ranking else None
            }
        
        return comparison_data
    
    def get_session_history(self) -> List[Dict[str, Any]]:
        """Get complete session history with energy metrics."""
        return [
            {
                'benchmark_name': session['benchmark_name'],
                'start_time': session['start_time'],
                'duration_seconds': session['duration_seconds'],
                'energy_kwh': session.get('carbon_footprint', {}).get('total_energy_kwh', 0),
                'carbon_kg': session.get('carbon_footprint', {}).get('carbon_emission_kg', 0),
                'milestones_count': len(session.get('milestones', [])),
                'metadata': session['metadata']
            }
            for session in self.benchmark_sessions
        ]
    
    def clear_history(self):
        """Clear all stored benchmark session history."""
        self.benchmark_sessions.clear()
        self.logger.info("Benchmark energy tracking history cleared")


class EnergyAwareScheduler:
    """
    Energy-aware benchmark scheduler that optimizes execution based on power consumption.
    """
    
    def __init__(self, energy_tracker: BenchmarkEnergyTracker):
        self.energy_tracker = energy_tracker
        self.logger = logging.getLogger(__name__)
        
        # Scheduling parameters
        self.max_power_threshold = 300  # Watts
        self.max_temperature_threshold = 80  # Celsius
        self.cooling_wait_time = 60  # seconds
    
    def should_execute_benchmark(self) -> tuple[bool, str]:
        """
        Determine if it's optimal to execute a benchmark based on current system state.
        
        Returns:
            Tuple of (should_execute, reason)
        """
        current_metrics = self.energy_tracker.get_real_time_metrics()
        
        if not current_metrics:
            return True, "Energy monitoring not available"
        
        current_measurement = current_metrics['current_measurement']
        
        # Check power consumption
        if current_measurement['system_power_watts']:
            if current_measurement['system_power_watts'] > self.max_power_threshold:
                return False, f"System power too high: {current_measurement['system_power_watts']:.1f}W > {self.max_power_threshold}W"
        
        # Check temperature
        if current_measurement['temperature_celsius']:
            if current_measurement['temperature_celsius'] > self.max_temperature_threshold:
                return False, f"System temperature too high: {current_measurement['temperature_celsius']:.1f}°C > {self.max_temperature_threshold}°C"
        
        # Check CPU usage
        if current_measurement['cpu_usage_percent'] > 90:
            return False, f"CPU usage too high: {current_measurement['cpu_usage_percent']:.1f}%"
        
        return True, "System ready for benchmark execution"
    
    def wait_for_optimal_conditions(self, timeout_seconds: int = 300) -> bool:
        """
        Wait for optimal system conditions before executing benchmark.
        
        Args:
            timeout_seconds: Maximum time to wait
            
        Returns:
            True if optimal conditions achieved, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            should_execute, reason = self.should_execute_benchmark()
            
            if should_execute:
                self.logger.info("Optimal conditions achieved for benchmark execution")
                return True
            
            self.logger.debug(f"Waiting for optimal conditions: {reason}")
            time.sleep(self.cooling_wait_time)
        
        self.logger.warning(f"Timeout waiting for optimal conditions after {timeout_seconds} seconds")
        return False
