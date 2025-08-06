#!/usr/bin/env python3
"""
ShadowBench - Main Benchmark Runner
Orchestrates adversarial AI benchmarking across multiple models and scenarios.
"""

import os
import sys
import json
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Import our core modules
from adversarial_injector.perturbation_engine import PerturbationEngine
from metrics.unified_scoring import UnifiedScoringFramework
from models.openai_wrapper import OpenAIWrapper
from models.anthropic_wrapper import AnthropicWrapper
from models.gemini_wrapper import GeminiWrapper
from models.llama_local import LlamaLocalWrapper


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    scenario: str
    model_configs: List[Dict[str, Any]]
    adversarial_config: Dict[str, Any]
    metrics_config: Dict[str, Any]
    output_dir: str = "results"
    max_retries: int = 3
    timeout: int = 30


class ShadowBenchRunner:
    """Main orchestration class for ShadowBench."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.logger = self._setup_logging()
        self.perturbation_engine = PerturbationEngine()
        self.scoring_framework = UnifiedScoringFramework()
        
        # Model wrappers
        self.model_wrappers = {
            'openai': OpenAIWrapper,
            'anthropic': AnthropicWrapper,
            'gemini': GeminiWrapper,
            'llama': LlamaLocalWrapper
        }
        
        self.results = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging."""
        logger = logging.getLogger('shadowbench')
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler
        os.makedirs('logs', exist_ok=True)
        file_handler = logging.FileHandler(f'logs/shadowbench_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def load_config(self, config_path: str) -> BenchmarkConfig:
        """Load benchmark configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                raw_config = yaml.safe_load(f)
            
            return BenchmarkConfig(**raw_config)
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            raise
    
    def initialize_model(self, model_config: Dict[str, Any]) -> Any:
        """Initialize a model wrapper based on configuration."""
        model_type = model_config.get('type')
        
        if model_type not in self.model_wrappers:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        try:
            wrapper_class = self.model_wrappers[model_type]
            return wrapper_class(**model_config.get('params', {}))
        except Exception as e:
            self.logger.error(f"Failed to initialize {model_type} model: {e}")
            raise
    
    def run_scenario(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Execute a complete benchmark scenario."""
        self.logger.info(f"Starting benchmark scenario: {config.scenario}")
        scenario_results = {
            'scenario': config.scenario,
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'adversarial_config': config.adversarial_config,
            'summary': {}
        }
        
        try:
            # Load test data
            test_data = self._load_test_data(config.scenario)
            
            # Process each model
            for model_config in config.model_configs:
                model_id = model_config.get('id', model_config['type'])
                self.logger.info(f"Processing model: {model_id}")
                
                try:
                    # Initialize model
                    model = self.initialize_model(model_config)
                    
                    # Run adversarial tests
                    model_results = self._run_adversarial_tests(
                        model, test_data, config.adversarial_config, config.metrics_config
                    )
                    
                    scenario_results['models'][model_id] = model_results
                    
                except Exception as e:
                    self.logger.error(f"Failed to process model {model_id}: {e}")
                    scenario_results['models'][model_id] = {'error': str(e)}
            
            # Generate unified scores
            scenario_results['summary'] = self.scoring_framework.calculate_unified_scores(
                scenario_results['models']
            )
            
            # Save results
            self._save_results(scenario_results, config.output_dir)
            
            self.logger.info(f"Completed benchmark scenario: {config.scenario}")
            return scenario_results
            
        except Exception as e:
            self.logger.error(f"Failed to run scenario {config.scenario}: {e}")
            raise
    
    def _load_test_data(self, scenario: str) -> List[Dict[str, Any]]:
        """Load test data for a scenario."""
        scenario_path = Path(f"scenarios/{scenario}")
        
        if not scenario_path.exists():
            raise FileNotFoundError(f"Scenario not found: {scenario}")
        
        # Look for data files
        data_files = list(scenario_path.glob("*.json")) + list(scenario_path.glob("*.jsonl"))
        
        if not data_files:
            raise FileNotFoundError(f"No data files found in {scenario_path}")
        
        test_data = []
        for data_file in data_files:
            with open(data_file, 'r') as f:
                if data_file.suffix == '.jsonl':
                    for line in f:
                        test_data.append(json.loads(line.strip()))
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        test_data.extend(data)
                    else:
                        test_data.append(data)
        
        return test_data
    
    def _run_adversarial_tests(self, model, test_data: List[Dict], 
                              adversarial_config: Dict, metrics_config: Dict) -> Dict[str, Any]:
        """Run adversarial tests on a model."""
        results = {
            'clean_performance': {},
            'adversarial_performance': {},
            'metrics': {},
            'test_count': len(test_data)
        }
        
        clean_responses = []
        adversarial_responses = []
        
        for i, test_case in enumerate(test_data):
            try:
                # Clean test
                clean_prompt = test_case.get('prompt', test_case.get('input', ''))
                clean_response = model.generate(clean_prompt)
                clean_responses.append({
                    'prompt': clean_prompt,
                    'response': clean_response,
                    'expected': test_case.get('expected', ''),
                    'metadata': test_case.get('metadata', {})
                })
                
                # Adversarial test
                adversarial_prompt = self.perturbation_engine.apply_perturbations(
                    clean_prompt, adversarial_config
                )
                adversarial_response = model.generate(adversarial_prompt)
                adversarial_responses.append({
                    'prompt': adversarial_prompt,
                    'response': adversarial_response,
                    'expected': test_case.get('expected', ''),
                    'metadata': test_case.get('metadata', {}),
                    'perturbations': adversarial_config
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to process test case {i}: {e}")
        
        # Calculate metrics
        results['metrics'] = self.scoring_framework.calculate_all_metrics(
            clean_responses, adversarial_responses, metrics_config
        )
        
        results['clean_performance'] = {
            'response_count': len(clean_responses),
            'success_rate': len(clean_responses) / len(test_data)
        }
        
        results['adversarial_performance'] = {
            'response_count': len(adversarial_responses),
            'success_rate': len(adversarial_responses) / len(test_data)
        }
        
        return results
    
    def _save_results(self, results: Dict[str, Any], output_dir: str):
        """Save benchmark results to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario_name = results['scenario'].replace('/', '_')
        filename = f"{scenario_name}_{timestamp}.json"
        filepath = Path(output_dir) / filename
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to: {filepath}")
    
    def run_benchmark(self, config_path: str) -> Dict[str, Any]:
        """Run a complete benchmark from configuration file."""
        config = self.load_config(config_path)
        return self.run_scenario(config)


def main():
    """Main entry point for ShadowBench CLI."""
    parser = argparse.ArgumentParser(description="ShadowBench - Adversarial AI Benchmarking Framework")
    parser.add_argument('--config', '-c', required=True, help='Path to configuration YAML file')
    parser.add_argument('--output', '-o', default='results', help='Output directory for results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger('shadowbench').setLevel(logging.DEBUG)
    
    try:
        runner = ShadowBenchRunner()
        results = runner.run_benchmark(args.config)
        
        print(f"\n{'='*60}")
        print("SHADOWBENCH RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Scenario: {results['scenario']}")
        print(f"Models tested: {len(results['models'])}")
        print(f"Timestamp: {results['timestamp']}")
        
        if 'summary' in results:
            print(f"\nUnified Scores:")
            for metric, score in results['summary'].items():
                if isinstance(score, (int, float)):
                    print(f"  {metric}: {score:.3f}")
        
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
