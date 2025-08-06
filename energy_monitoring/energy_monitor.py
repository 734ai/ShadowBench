"""
Real-time Energy Monitoring and Carbon Footprint Tracking for ShadowBench
Monitors GPU, CPU, and system-wide power consumption with sustainability metrics.
"""

import time
import psutil
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import os
import subprocess
import platform


@dataclass
class EnergyMeasurement:
    """Single energy measurement data point."""
    timestamp: float
    cpu_usage_percent: float
    cpu_power_watts: Optional[float]
    gpu_usage_percent: Optional[float]
    gpu_power_watts: Optional[float]
    memory_usage_mb: float
    disk_io_mb_per_sec: float
    network_io_mb_per_sec: float
    system_power_watts: Optional[float]
    temperature_celsius: Optional[float]


@dataclass
class CarbonFootprint:
    """Carbon footprint calculation results."""
    total_energy_kwh: float
    carbon_emission_kg: float
    carbon_intensity_factor: float  # kg CO2 per kWh
    equivalent_metrics: Dict[str, float]  # Car miles, tree-months, etc.


class EnergyMonitor:
    """
    Real-time system energy monitoring with carbon footprint analysis.
    
    Features:
    - Multi-platform CPU/GPU power monitoring
    - Real-time performance tracking
    - Carbon footprint calculation
    - Energy efficiency optimization suggestions
    - Historical trend analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Monitoring configuration
        self.sampling_interval = self.config.get('sampling_interval', 1.0)  # seconds
        self.max_history_points = self.config.get('max_history', 3600)  # 1 hour at 1Hz
        self.carbon_intensity = self.config.get('carbon_intensity', 0.233)  # kg CO2/kWh (US avg)
        
        # Data storage
        self.measurements = deque(maxlen=self.max_history_points)
        self.session_start_time = time.time()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.callbacks: List[Callable[[EnergyMeasurement], None]] = []
        
        # Hardware detection
        self.has_nvidia_gpu = self._detect_nvidia_gpu()
        self.has_intel_gpu = self._detect_intel_gpu()
        self.has_amd_gpu = self._detect_amd_gpu()
        
        # Power measurement tools
        self.power_tools = self._initialize_power_tools()
        
        # Baseline measurements for comparison
        self.baseline_measurement = None
        
        self.logger.info(f"EnergyMonitor initialized with {len(self.power_tools)} power monitoring tools")
    
    def start_monitoring(self) -> bool:
        """Start continuous energy monitoring."""
        if self.is_monitoring:
            self.logger.warning("Energy monitoring already active")
            return True
        
        try:
            self.is_monitoring = True
            self.session_start_time = time.time()
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            
            # Take baseline measurement
            self.baseline_measurement = self._take_measurement()
            
            self.logger.info("Energy monitoring started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start energy monitoring: {e}")
            self.is_monitoring = False
            return False
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop energy monitoring and return session summary."""
        if not self.is_monitoring:
            return {'error': 'Monitoring not active'}
        
        self.is_monitoring = False
        
        # Wait for monitoring thread to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        # Calculate session summary
        session_summary = self._calculate_session_summary()
        
        self.logger.info("Energy monitoring stopped")
        return session_summary
    
    def get_current_measurement(self) -> Optional[EnergyMeasurement]:
        """Get current energy measurement."""
        try:
            return self._take_measurement()
        except Exception as e:
            self.logger.error(f"Failed to get current measurement: {e}")
            return None
    
    def get_historical_data(self, duration_seconds: Optional[int] = None) -> List[EnergyMeasurement]:
        """Get historical measurement data."""
        if duration_seconds is None:
            return list(self.measurements)
        
        cutoff_time = time.time() - duration_seconds
        return [m for m in self.measurements if m.timestamp >= cutoff_time]
    
    def calculate_carbon_footprint(self, duration_seconds: Optional[int] = None) -> CarbonFootprint:
        """Calculate carbon footprint for specified duration."""
        measurements = self.get_historical_data(duration_seconds)
        
        if not measurements:
            return CarbonFootprint(0, 0, self.carbon_intensity, {})
        
        # Calculate total energy consumption
        total_energy_wh = 0
        prev_measurement = None
        
        for measurement in measurements:
            if prev_measurement is not None:
                time_delta_hours = (measurement.timestamp - prev_measurement.timestamp) / 3600
                
                # Estimate power consumption
                avg_power = self._estimate_average_power(prev_measurement, measurement)
                energy_increment = avg_power * time_delta_hours
                total_energy_wh += energy_increment
            
            prev_measurement = measurement
        
        total_energy_kwh = total_energy_wh / 1000
        carbon_emission_kg = total_energy_kwh * self.carbon_intensity
        
        # Calculate equivalent metrics
        equivalent_metrics = {
            'car_miles_equivalent': carbon_emission_kg / 0.404,  # kg CO2 per mile
            'tree_months_to_offset': carbon_emission_kg / 0.025,  # kg CO2 per tree per month
            'smartphone_charges': total_energy_kwh / 0.01,  # kWh per charge
            'light_bulb_hours': total_energy_kwh / 0.01,  # 10W LED bulb
        }
        
        return CarbonFootprint(
            total_energy_kwh=total_energy_kwh,
            carbon_emission_kg=carbon_emission_kg,
            carbon_intensity_factor=self.carbon_intensity,
            equivalent_metrics=equivalent_metrics
        )
    
    def add_callback(self, callback: Callable[[EnergyMeasurement], None]):
        """Add callback for real-time measurement updates."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[EnergyMeasurement], None]):
        """Remove measurement callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def get_efficiency_recommendations(self) -> List[Dict[str, Any]]:
        """Get energy efficiency optimization recommendations."""
        if len(self.measurements) < 10:
            return [{'recommendation': 'Insufficient data for analysis', 'priority': 'info'}]
        
        recommendations = []
        recent_measurements = list(self.measurements)[-60:]  # Last minute
        
        # Analyze CPU usage patterns
        avg_cpu = sum(m.cpu_usage_percent for m in recent_measurements) / len(recent_measurements)
        if avg_cpu > 80:
            recommendations.append({
                'recommendation': 'High CPU usage detected. Consider optimizing algorithms or reducing batch size.',
                'priority': 'high',
                'metric': f'Average CPU: {avg_cpu:.1f}%'
            })
        
        # Analyze GPU usage patterns
        gpu_measurements = [m for m in recent_measurements if m.gpu_usage_percent is not None]
        if gpu_measurements:
            avg_gpu = sum(m.gpu_usage_percent for m in gpu_measurements) / len(gpu_measurements)
            if avg_gpu < 30:
                recommendations.append({
                    'recommendation': 'Low GPU utilization. Consider increasing batch size or using CPU-only mode.',
                    'priority': 'medium',
                    'metric': f'Average GPU: {avg_gpu:.1f}%'
                })
            elif avg_gpu > 95:
                recommendations.append({
                    'recommendation': 'GPU at maximum utilization. Consider distributed computing or model optimization.',
                    'priority': 'medium',
                    'metric': f'Average GPU: {avg_gpu:.1f}%'
                })
        
        # Analyze memory usage
        avg_memory = sum(m.memory_usage_mb for m in recent_measurements) / len(recent_measurements)
        total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
        memory_usage_percent = (avg_memory / total_memory_mb) * 100
        
        if memory_usage_percent > 85:
            recommendations.append({
                'recommendation': 'High memory usage detected. Consider reducing model size or batch size.',
                'priority': 'high',
                'metric': f'Memory usage: {memory_usage_percent:.1f}%'
            })
        
        # Power efficiency analysis
        power_measurements = [m for m in recent_measurements if m.system_power_watts is not None]
        if power_measurements and len(power_measurements) > 5:
            power_trend = self._analyze_power_trend(power_measurements)
            if power_trend > 0.1:  # Increasing power consumption
                recommendations.append({
                    'recommendation': 'Power consumption is increasing. Check for resource leaks or inefficient operations.',
                    'priority': 'medium',
                    'metric': f'Power trend: +{power_trend:.2f}W/min'
                })
        
        if not recommendations:
            recommendations.append({
                'recommendation': 'System operating efficiently within normal parameters.',
                'priority': 'info',
                'metric': 'All metrics within optimal ranges'
            })
        
        return recommendations
    
    def export_data(self, format: str = 'json', filepath: Optional[str] = None) -> str:
        """Export monitoring data to file."""
        data = {
            'session_info': {
                'start_time': self.session_start_time,
                'end_time': time.time(),
                'duration_seconds': time.time() - self.session_start_time,
                'measurement_count': len(self.measurements)
            },
            'configuration': {
                'sampling_interval': self.sampling_interval,
                'carbon_intensity': self.carbon_intensity,
                'hardware_detected': {
                    'nvidia_gpu': self.has_nvidia_gpu,
                    'intel_gpu': self.has_intel_gpu,
                    'amd_gpu': self.has_amd_gpu
                }
            },
            'measurements': [asdict(m) for m in self.measurements],
            'carbon_footprint': asdict(self.calculate_carbon_footprint()),
            'efficiency_recommendations': self.get_efficiency_recommendations()
        }
        
        if format.lower() == 'json':
            json_data = json.dumps(data, indent=2)
            if filepath:
                with open(filepath, 'w') as f:
                    f.write(json_data)
                self.logger.info(f"Energy data exported to {filepath}")
            return json_data
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in separate thread."""
        while self.is_monitoring:
            try:
                measurement = self._take_measurement()
                self.measurements.append(measurement)
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(measurement)
                    except Exception as e:
                        self.logger.warning(f"Callback error: {e}")
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.sampling_interval)
    
    def _take_measurement(self) -> EnergyMeasurement:
        """Take single energy measurement."""
        timestamp = time.time()
        
        # CPU metrics
        cpu_usage = psutil.cpu_percent(interval=None)
        cpu_power = self._measure_cpu_power()
        
        # GPU metrics
        gpu_usage, gpu_power = self._measure_gpu_metrics()
        
        # Memory metrics
        memory_info = psutil.virtual_memory()
        memory_usage_mb = (memory_info.total - memory_info.available) / (1024 * 1024)
        
        # I/O metrics
        disk_io = self._measure_disk_io()
        network_io = self._measure_network_io()
        
        # System power and temperature
        system_power = self._measure_system_power()
        temperature = self._measure_temperature()
        
        return EnergyMeasurement(
            timestamp=timestamp,
            cpu_usage_percent=cpu_usage,
            cpu_power_watts=cpu_power,
            gpu_usage_percent=gpu_usage,
            gpu_power_watts=gpu_power,
            memory_usage_mb=memory_usage_mb,
            disk_io_mb_per_sec=disk_io,
            network_io_mb_per_sec=network_io,
            system_power_watts=system_power,
            temperature_celsius=temperature
        )
    
    def _detect_nvidia_gpu(self) -> bool:
        """Detect NVIDIA GPU presence."""
        try:
            subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv'], 
                         capture_output=True, check=True, timeout=5)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _detect_intel_gpu(self) -> bool:
        """Detect Intel GPU presence."""
        try:
            # Check for Intel GPU on Linux
            if platform.system() == 'Linux':
                result = subprocess.run(['lspci', '-nn'], capture_output=True, text=True)
                return 'Intel' in result.stdout and ('VGA' in result.stdout or 'Display' in result.stdout)
            return False
        except:
            return False
    
    def _detect_amd_gpu(self) -> bool:
        """Detect AMD GPU presence."""
        try:
            # Check for AMD GPU
            if platform.system() == 'Linux':
                result = subprocess.run(['lspci', '-nn'], capture_output=True, text=True)
                return 'AMD' in result.stdout and ('VGA' in result.stdout or 'Display' in result.stdout)
            return False
        except:
            return False
    
    def _initialize_power_tools(self) -> List[str]:
        """Initialize available power measurement tools."""
        tools = []
        
        # Check for various power monitoring tools
        tool_commands = {
            'nvidia-smi': ['nvidia-smi', '--help'],
            'powertop': ['powertop', '--version'],
            'turbostat': ['turbostat', '--version'],
            'sensors': ['sensors', '--version'],
            'powerstat': ['powerstat', '--version']
        }
        
        for tool_name, test_command in tool_commands.items():
            try:
                subprocess.run(test_command, capture_output=True, check=True, timeout=3)
                tools.append(tool_name)
            except:
                pass
        
        return tools
    
    def _measure_cpu_power(self) -> Optional[float]:
        """Measure CPU power consumption."""
        try:
            # Try different methods based on available tools
            if 'turbostat' in self.power_tools:
                return self._get_turbostat_cpu_power()
            elif platform.system() == 'Linux':
                return self._get_linux_cpu_power()
            else:
                # Estimate based on CPU usage and TDP
                cpu_usage = psutil.cpu_percent()
                estimated_tdp = 65  # Watts (typical desktop CPU)
                return (cpu_usage / 100) * estimated_tdp
        except:
            return None
    
    def _measure_gpu_metrics(self) -> tuple[Optional[float], Optional[float]]:
        """Measure GPU usage and power consumption."""
        gpu_usage = None
        gpu_power = None
        
        try:
            if self.has_nvidia_gpu:
                gpu_usage, gpu_power = self._get_nvidia_metrics()
            # Add support for other GPU vendors as needed
        except:
            pass
        
        return gpu_usage, gpu_power
    
    def _get_nvidia_metrics(self) -> tuple[float, float]:
        """Get NVIDIA GPU metrics using nvidia-smi."""
        cmd = [
            'nvidia-smi',
            '--query-gpu=utilization.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
        lines = result.stdout.strip().split('\n')
        
        # Parse first GPU (can be extended for multi-GPU)
        if lines and lines[0]:
            parts = lines[0].split(',')
            usage = float(parts[0].strip())
            power = float(parts[1].strip()) if parts[1].strip() != 'N/A' else 0.0
            return usage, power
        
        return 0.0, 0.0
    
    def _measure_disk_io(self) -> float:
        """Measure disk I/O in MB/s."""
        try:
            disk_io_before = psutil.disk_io_counters()
            time.sleep(0.1)  # Brief sampling period
            disk_io_after = psutil.disk_io_counters()
            
            if disk_io_before and disk_io_after:
                bytes_delta = (disk_io_after.read_bytes + disk_io_after.write_bytes) - \
                            (disk_io_before.read_bytes + disk_io_before.write_bytes)
                mb_per_sec = (bytes_delta / (1024 * 1024)) / 0.1
                return mb_per_sec
        except:
            pass
        
        return 0.0
    
    def _measure_network_io(self) -> float:
        """Measure network I/O in MB/s."""
        try:
            net_io_before = psutil.net_io_counters()
            time.sleep(0.1)  # Brief sampling period
            net_io_after = psutil.net_io_counters()
            
            if net_io_before and net_io_after:
                bytes_delta = (net_io_after.bytes_sent + net_io_after.bytes_recv) - \
                            (net_io_before.bytes_sent + net_io_before.bytes_recv)
                mb_per_sec = (bytes_delta / (1024 * 1024)) / 0.1
                return mb_per_sec
        except:
            pass
        
        return 0.0
    
    def _measure_system_power(self) -> Optional[float]:
        """Measure total system power consumption."""
        try:
            # Try different methods based on available tools
            if 'powerstat' in self.power_tools:
                return self._get_powerstat_power()
            elif platform.system() == 'Linux':
                return self._get_linux_system_power()
            else:
                # Estimate based on component power
                cpu_power = self._measure_cpu_power() or 0
                gpu_power = self._measure_gpu_metrics()[1] or 0
                base_power = 50  # Motherboard, RAM, etc.
                return cpu_power + gpu_power + base_power
        except:
            return None
    
    def _measure_temperature(self) -> Optional[float]:
        """Measure system temperature."""
        try:
            if 'sensors' in self.power_tools:
                return self._get_sensors_temperature()
            elif platform.system() == 'Linux':
                return self._get_linux_temperature()
        except:
            pass
        
        return None
    
    def _get_turbostat_cpu_power(self) -> float:
        """Get CPU power using turbostat."""
        # Simplified implementation - would need proper turbostat parsing
        return None
    
    def _get_linux_cpu_power(self) -> Optional[float]:
        """Get CPU power on Linux systems."""
        try:
            # Try RAPL (Running Average Power Limit) interface
            rapl_path = '/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj'
            if os.path.exists(rapl_path):
                with open(rapl_path, 'r') as f:
                    energy_uj = int(f.read().strip())
                # This would need time-based calculation for actual power
                return None  # Placeholder for proper implementation
        except:
            pass
        
        return None
    
    def _get_powerstat_power(self) -> Optional[float]:
        """Get system power using powerstat."""
        # Implementation would parse powerstat output
        return None
    
    def _get_linux_system_power(self) -> Optional[float]:
        """Get system power on Linux."""
        # Implementation would read from power supply interfaces
        return None
    
    def _get_sensors_temperature(self) -> Optional[float]:
        """Get temperature using sensors."""
        try:
            result = subprocess.run(['sensors', '-u'], capture_output=True, text=True, timeout=3)
            # Parse sensors output for CPU temperature
            for line in result.stdout.split('\n'):
                if 'temp1_input' in line or 'Core 0' in line:
                    temp_value = float(line.split(':')[1].strip())
                    return temp_value
        except:
            pass
        
        return None
    
    def _get_linux_temperature(self) -> Optional[float]:
        """Get temperature on Linux systems."""
        try:
            thermal_paths = [
                '/sys/class/thermal/thermal_zone0/temp',
                '/sys/class/thermal/thermal_zone1/temp'
            ]
            
            for path in thermal_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        temp_millic = int(f.read().strip())
                        return temp_millic / 1000.0
        except:
            pass
        
        return None
    
    def _estimate_average_power(self, measurement1: EnergyMeasurement, 
                              measurement2: EnergyMeasurement) -> float:
        """Estimate average power between two measurements."""
        power1 = measurement1.system_power_watts or 0
        power2 = measurement2.system_power_watts or 0
        
        if power1 > 0 and power2 > 0:
            return (power1 + power2) / 2
        elif power1 > 0:
            return power1
        elif power2 > 0:
            return power2
        else:
            # Estimate based on CPU and GPU usage
            cpu_power = (measurement1.cpu_usage_percent + measurement2.cpu_usage_percent) / 200 * 65
            
            gpu_power1 = measurement1.gpu_power_watts or 0
            gpu_power2 = measurement2.gpu_power_watts or 0
            gpu_power = (gpu_power1 + gpu_power2) / 2
            
            return cpu_power + gpu_power + 50  # Base system power
    
    def _analyze_power_trend(self, measurements: List[EnergyMeasurement]) -> float:
        """Analyze power consumption trend (watts per minute)."""
        if len(measurements) < 2:
            return 0.0
        
        times = [m.timestamp for m in measurements]
        powers = [m.system_power_watts or 0 for m in measurements]
        
        # Simple linear trend analysis
        time_range = times[-1] - times[0]
        power_change = powers[-1] - powers[0]
        
        if time_range > 0:
            return (power_change / time_range) * 60  # Convert to per minute
        
        return 0.0
    
    def _calculate_session_summary(self) -> Dict[str, Any]:
        """Calculate comprehensive session summary."""
        if not self.measurements:
            return {'error': 'No measurements available'}
        
        duration = time.time() - self.session_start_time
        carbon_footprint = self.calculate_carbon_footprint()
        
        # Calculate averages and peaks
        cpu_usages = [m.cpu_usage_percent for m in self.measurements]
        memory_usages = [m.memory_usage_mb for m in self.measurements]
        gpu_usages = [m.gpu_usage_percent for m in self.measurements if m.gpu_usage_percent is not None]
        
        summary = {
            'session_duration_seconds': duration,
            'total_measurements': len(self.measurements),
            'energy_metrics': {
                'total_energy_kwh': carbon_footprint.total_energy_kwh,
                'average_power_watts': self._calculate_average_power(),
                'peak_power_watts': self._calculate_peak_power()
            },
            'resource_utilization': {
                'cpu': {
                    'average_percent': sum(cpu_usages) / len(cpu_usages),
                    'peak_percent': max(cpu_usages),
                    'min_percent': min(cpu_usages)
                },
                'memory': {
                    'average_mb': sum(memory_usages) / len(memory_usages),
                    'peak_mb': max(memory_usages),
                    'min_mb': min(memory_usages)
                }
            },
            'carbon_footprint': asdict(carbon_footprint),
            'efficiency_recommendations': self.get_efficiency_recommendations()
        }
        
        if gpu_usages:
            summary['resource_utilization']['gpu'] = {
                'average_percent': sum(gpu_usages) / len(gpu_usages),
                'peak_percent': max(gpu_usages),
                'min_percent': min(gpu_usages)
            }
        
        return summary
    
    def _calculate_average_power(self) -> Optional[float]:
        """Calculate average power consumption."""
        power_measurements = [m.system_power_watts for m in self.measurements if m.system_power_watts is not None]
        if power_measurements:
            return sum(power_measurements) / len(power_measurements)
        return None
    
    def _calculate_peak_power(self) -> Optional[float]:
        """Calculate peak power consumption."""
        power_measurements = [m.system_power_watts for m in self.measurements if m.system_power_watts is not None]
        if power_measurements:
            return max(power_measurements)
        return None
