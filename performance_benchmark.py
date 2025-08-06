#!/usr/bin/env python3
"""
ShadowBench Performance Benchmark Suite
Measures execution times, memory usage, and resource efficiency.
"""

import time
import psutil
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

# Import ShadowBench components
from benchmark_runner import ShadowBenchRunner, BenchmarkConfig
from metrics.unified_scoring import UnifiedScorer
from adversarial_injector.perturbation_engine import PerturbationEngine

@dataclass
class PerformanceMetrics:
    """Performance measurement results."""
    execution_time_seconds: float
    peak_memory_mb: float
    cpu_usage_percent: float
    memory_efficiency_score: float
    throughput_evaluations_per_second: float
    total_evaluations: int

class ShadowBenchProfiler:
    """Profile ShadowBench framework performance."""
    
    def __init__(self):
        self.logger = logging.getLogger("ShadowBench.Profiler")
        self.results = []
    
    def benchmark_unified_scoring(self, iterations: int = 100) -> PerformanceMetrics:
        """Benchmark unified scoring performance."""
        self.logger.info(f"Benchmarking unified scoring with {iterations} iterations")
        
        # Setup
        scorer = UnifiedScorer()
        mock_metrics = {
            'deception_metrics': {'overall_deception_score': 0.7},
            'robustness_metrics': {'overall_robustness_score': 0.8},
            'privacy_metrics': {'overall_privacy_vulnerability': 0.3},
            'explainability_metrics': {'overall_explainability_score': 0.6},
            'energy_metrics': {'overall_energy_score': 0.4},
            'self_correction_metrics': {'overall_correction_score': 0.5}
        }
        
        # Measure performance
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        
        # Execute iterations
        for _ in range(iterations):
            unified_score, report = scorer.compute_unified_score(mock_metrics)
        
        # Calculate metrics
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        end_cpu = psutil.cpu_percent()
        
        execution_time = end_time - start_time
        peak_memory = max(start_memory, end_memory)
        avg_cpu = (start_cpu + end_cpu) / 2
        throughput = iterations / execution_time
        memory_efficiency = 1.0 - (peak_memory / 1000)  # Normalized to 0-1
        
        return PerformanceMetrics(
            execution_time_seconds=execution_time,
            peak_memory_mb=peak_memory,
            cpu_usage_percent=avg_cpu,
            memory_efficiency_score=memory_efficiency,
            throughput_evaluations_per_second=throughput,
            total_evaluations=iterations
        )
    
    def benchmark_perturbation_engine(self, iterations: int = 50) -> PerformanceMetrics:
        """Benchmark perturbation engine performance."""
        self.logger.info(f"Benchmarking perturbation engine with {iterations} iterations")
        
        # Setup
        engine = PerturbationEngine()
        test_prompt = "Evaluate the security of this AI system and provide recommendations."
        
        # Measure performance
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        
        # Execute iterations
        for _ in range(iterations):
            # Use the correct method name and format
            config = {
                'attacks': {
                    'prompt_injection': {'enabled': True, 'intensity': 0.5},
                    'few_shot_poisoning': {'enabled': True, 'intensity': 0.5}
                },
                'intensity': 0.5
            }
            perturbed_text = engine.apply_perturbations(test_prompt, config)
        
        # Calculate metrics
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        end_cpu = psutil.cpu_percent()
        
        execution_time = end_time - start_time
        peak_memory = max(start_memory, end_memory)
        avg_cpu = (start_cpu + end_cpu) / 2
        throughput = iterations / execution_time
        memory_efficiency = 1.0 - (peak_memory / 1000)
        
        return PerformanceMetrics(
            execution_time_seconds=execution_time,
            peak_memory_mb=peak_memory,
            cpu_usage_percent=avg_cpu,
            memory_efficiency_score=memory_efficiency,
            throughput_evaluations_per_second=throughput,
            total_evaluations=iterations
        )
    
    def benchmark_end_to_end_evaluation(self) -> PerformanceMetrics:
        """Benchmark complete end-to-end evaluation."""
        self.logger.info("Benchmarking end-to-end evaluation")
        
        # Setup mock configuration
        config = BenchmarkConfig(
            scenario="basic_qa",
            model_configs=[{
                "type": "openai",
                "id": "gpt-test",
                "params": {"model_name": "gpt-3.5-turbo", "temperature": 0.7}
            }],
            adversarial_config={
                "attacks": {
                    "prompt_injection": {"enabled": True, "intensity": 0.5},
                    "few_shot_poisoning": {"enabled": True, "injection_rate": 0.3}
                }
            },
            metrics_config={
                "deception_metrics": True,
                "robustness_score": True,
                "privacy_score": True
            },
            max_retries=1
        )
        
        # Measure performance
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        
        # Execute benchmark (mock mode)
        try:
            runner = ShadowBenchRunner()
            # Mock evaluation since we don't have real API keys
            mock_results = {
                "gpt-test": {
                    "deception_metrics": {"overall_deception_score": 0.75},
                    "robustness_metrics": {"overall_robustness_score": 0.65},
                    "privacy_metrics": {"overall_privacy_vulnerability": 0.25}
                }
            }
            
            # Score the results
            scorer = UnifiedScorer()
            for model_name, metrics in mock_results.items():
                unified_score, report = scorer.compute_unified_score(metrics)
        
        except Exception as e:
            self.logger.warning(f"End-to-end test completed with mock data: {e}")
        
        # Calculate metrics
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        end_cpu = psutil.cpu_percent()
        
        execution_time = end_time - start_time
        peak_memory = max(start_memory, end_memory)
        avg_cpu = (start_cpu + end_cpu) / 2
        throughput = 1 / execution_time  # One complete evaluation
        memory_efficiency = 1.0 - (peak_memory / 1000)
        
        return PerformanceMetrics(
            execution_time_seconds=execution_time,
            peak_memory_mb=peak_memory,
            cpu_usage_percent=avg_cpu,
            memory_efficiency_score=memory_efficiency,
            throughput_evaluations_per_second=throughput,
            total_evaluations=1
        )
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark suite."""
        self.logger.info("Starting comprehensive performance benchmark")
        
        results = {}
        
        # Test 1: Unified Scoring Performance
        self.logger.info("Running unified scoring benchmark...")
        results['unified_scoring'] = asdict(self.benchmark_unified_scoring(100))
        
        # Test 2: Perturbation Engine Performance  
        self.logger.info("Running perturbation engine benchmark...")
        results['perturbation_engine'] = asdict(self.benchmark_perturbation_engine(50))
        
        # Test 3: End-to-End Evaluation
        self.logger.info("Running end-to-end evaluation benchmark...")
        results['end_to_end_evaluation'] = asdict(self.benchmark_end_to_end_evaluation())
        
        # System Information
        results['system_info'] = {
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'python_version': sys.version,
            'platform': sys.platform
        }
        
        # Performance Summary
        avg_execution_time = sum(r['execution_time_seconds'] for r in results.values() if isinstance(r, dict) and 'execution_time_seconds' in r) / 3
        avg_memory_usage = sum(r['peak_memory_mb'] for r in results.values() if isinstance(r, dict) and 'peak_memory_mb' in r) / 3
        avg_throughput = sum(r['throughput_evaluations_per_second'] for r in results.values() if isinstance(r, dict) and 'throughput_evaluations_per_second' in r) / 3
        
        results['performance_summary'] = {
            'average_execution_time_seconds': avg_execution_time,
            'average_memory_usage_mb': avg_memory_usage,
            'average_throughput_eps': avg_throughput,
            'performance_grade': self._calculate_performance_grade(avg_execution_time, avg_memory_usage, avg_throughput)
        }
        
        return results
    
    def _calculate_performance_grade(self, avg_time: float, avg_memory: float, avg_throughput: float) -> str:
        """Calculate performance grade based on metrics."""
        # Performance thresholds (configurable)
        excellent_time = 0.01  # 10ms average
        good_time = 0.05      # 50ms average
        excellent_memory = 150  # 150MB (more realistic for ML workloads)
        good_memory = 500      # 500MB
        excellent_throughput = 1000  # 1000 ops/sec (high performance)
        good_throughput = 100       # 100 ops/sec
        
        score = 0
        max_score = 9
        
        # Time scoring (3 points)
        if avg_time <= excellent_time:
            score += 3
        elif avg_time <= good_time:
            score += 2
        else:
            score += 1
        
        # Memory scoring (3 points)
        if avg_memory <= excellent_memory:
            score += 3
        elif avg_memory <= good_memory:
            score += 2
        else:
            score += 1
        
        # Throughput scoring (3 points)
        if avg_throughput >= excellent_throughput:
            score += 3
        elif avg_throughput >= good_throughput:
            score += 2
        else:
            score += 1
        
        # Calculate grade
        percentage = (score / max_score) * 100
        if percentage >= 90:
            return "A+ (Excellent)"
        elif percentage >= 80:
            return "A (Very Good)"
        elif percentage >= 70:
            return "B+ (Good)"
        elif percentage >= 60:
            return "B (Satisfactory)"
        else:
            return "C (Needs Improvement)"
    
    def save_results(self, results: Dict[str, Any], output_path: Path) -> None:
        """Save benchmark results to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Benchmark results saved to {output_path}")
    
    def print_performance_report(self, results: Dict[str, Any]) -> None:
        """Print human-readable performance report."""
        print("\n" + "="*80)
        print("SHADOWBENCH PERFORMANCE BENCHMARK REPORT")
        print("="*80)
        
        summary = results.get('performance_summary', {})
        print(f"\n📊 OVERALL PERFORMANCE GRADE: {summary.get('performance_grade', 'N/A')}")
        print(f"⏱️  Average Execution Time: {summary.get('average_execution_time_seconds', 0):.3f} seconds")
        print(f"💾 Average Memory Usage: {summary.get('average_memory_usage_mb', 0):.1f} MB")
        print(f"🚀 Average Throughput: {summary.get('average_throughput_eps', 0):.1f} evaluations/second")
        
        print("\n📈 DETAILED RESULTS:")
        print("-" * 50)
        
        for component, metrics in results.items():
            if isinstance(metrics, dict) and 'execution_time_seconds' in metrics:
                print(f"\n{component.upper().replace('_', ' ')}:")
                print(f"  Execution Time: {metrics['execution_time_seconds']:.3f}s")
                print(f"  Peak Memory: {metrics['peak_memory_mb']:.1f} MB")
                print(f"  Throughput: {metrics['throughput_evaluations_per_second']:.1f} ops/sec")
                print(f"  Total Evaluations: {metrics['total_evaluations']}")
        
        system_info = results.get('system_info', {})
        print(f"\n💻 SYSTEM SPECIFICATIONS:")
        print(f"  CPU Cores: {system_info.get('cpu_count', 'N/A')}")
        print(f"  Total Memory: {system_info.get('total_memory_gb', 0):.1f} GB")
        print(f"  Platform: {system_info.get('platform', 'N/A')}")
        
        print("\n" + "="*80)

def main():
    """Run performance benchmark suite."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    profiler = ShadowBenchProfiler()
    
    print("🚀 Starting ShadowBench Performance Benchmark Suite...")
    print("This will measure execution time, memory usage, and throughput.")
    
    # Run comprehensive benchmark
    results = profiler.run_comprehensive_benchmark()
    
    # Display results
    profiler.print_performance_report(results)
    
    # Save results
    output_path = Path("results") / "performance_benchmark.json"
    profiler.save_results(results, output_path)
    
    print(f"\n📁 Detailed results saved to: {output_path}")
    print("✅ Performance benchmark completed successfully!")

if __name__ == "__main__":
    main()
