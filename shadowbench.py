#!/usr/bin/env python3
"""
ShadowBench CLI - Command Line Interface
Provides comprehensive CLI for running adversarial AI benchmarks.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_runner import ShadowBenchRunner, BenchmarkConfig


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def run_benchmark_command(args):
    """Run a benchmark from configuration file."""
    print(f"\nShadowBench - Running Adversarial AI Evaluation")
    print(f"{'='*60}")
    
    try:
        runner = ShadowBenchRunner()
        results = runner.run_benchmark(args.config)
        
        # Display results summary
        print(f"\nBenchmark completed successfully!")
        print(f"Scenario: {results['scenario']}")
        print(f"Models evaluated: {len(results['models'])}")
        
        if 'summary' in results:
            print(f"\nUnified Scores Summary:")
            for model_id, model_results in results['models'].items():
                if 'error' not in model_results:
                    print(f"  {model_id}: Overall security score pending")
        
        # Save detailed results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            results_file = output_path / f"{results['scenario'].replace('/', '_')}_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\nDetailed results saved to: {results_file}")
        
        return 0
        
    except Exception as e:
        print(f"\nError running benchmark: {e}")
        return 1


def list_scenarios_command(args):
    """List available scenarios."""
    scenarios_dir = Path("scenarios")
    
    print(f"\nAvailable ShadowBench Scenarios:")
    print(f"{'='*60}")
    
    if not scenarios_dir.exists():
        print("No scenarios directory found.")
        return 1
    
    for category_dir in scenarios_dir.iterdir():
        if category_dir.is_dir():
            print(f"\n{category_dir.name.upper()} Scenarios:")
            
            for scenario_file in category_dir.glob("*.py"):
                if scenario_file.name != "__init__.py":
                    scenario_name = scenario_file.stem
                    print(f"  - {category_dir.name}/{scenario_name}")
    
    return 0


def validate_config_command(args):
    """Validate a configuration file."""
    print(f"\nValidating configuration: {args.config}")
    print(f"{'='*60}")
    
    try:
        with open(args.config, 'r') as f:
            import yaml
            config_data = yaml.safe_load(f)
        
        # Basic validation
        required_fields = ['scenario', 'model_configs', 'adversarial_config']
        missing_fields = [field for field in required_fields if field not in config_data]
        
        if missing_fields:
            print(f"Missing required fields: {', '.join(missing_fields)}")
            return 1
        
        print(f"Configuration is valid!")
        print(f"  Scenario: {config_data['scenario']}")
        print(f"  Models: {len(config_data['model_configs'])}")
        print(f"  Attack types: {len(config_data.get('adversarial_config', {}).get('attacks', {}))}")
        
        return 0
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return 1


def create_config_command(args):
    """Create a sample configuration file."""
    config_template = {
        'scenario': 'llm/basic_qa',
        'model_configs': [
            {
                'id': 'gpt-3.5',
                'type': 'openai',
                'params': {
                    'model_name': 'gpt-3.5-turbo',
                    'max_tokens': 500,
                    'temperature': 0.7
                }
            }
        ],
        'adversarial_config': {
            'attacks': {
                'prompt_injection': {
                    'enabled': True,
                    'type': 'prefix',
                    'intensity': 0.5
                },
                'few_shot_poisoning': {
                    'enabled': True,
                    'injection_rate': 0.3
                }
            }
        },
        'metrics_config': {
            'deception_metrics': True,
            'robustness_score': True,
            'privacy_score': True
        },
        'output_dir': 'results',
        'max_retries': 3,
        'timeout': 30
    }
    
    output_file = args.output or 'shadowbench_config.yaml'
    
    try:
        import yaml
        with open(output_file, 'w') as f:
            yaml.dump(config_template, f, default_flow_style=False, indent=2)
        
        print(f"Sample configuration created: {output_file}")
        print(f"\nEdit the configuration file to customize your evaluation:")
        print(f"  - Add/remove models in 'model_configs'")
        print(f"  - Enable/disable attacks in 'adversarial_config'")
        print(f"  - Configure API keys in environment variables")
        
        return 0
        
    except Exception as e:
        print(f"Failed to create configuration: {e}")
        return 1


def test_models_command(args):
    """Test model connections."""
    print(f"\nTesting Model Connections")
    print(f"{'='*60}")
    
    from models import OpenAIWrapper, AnthropicWrapper, GeminiWrapper, LlamaLocalWrapper
    
    # Test configurations
    test_configs = [
        ('OpenAI', OpenAIWrapper, {}),
        ('Anthropic', AnthropicWrapper, {}),
        ('Google Gemini', GeminiWrapper, {}),
    ]
    
    results = {}
    
    for name, wrapper_class, params in test_configs:
        print(f"\nTesting {name}...")
        try:
            wrapper = wrapper_class(**params)
            if wrapper.validate_connection():
                print(f"  {name}: Connection successful")
                results[name] = True
            else:
                print(f"  {name}: Connection failed")
                results[name] = False
        except Exception as e:
            print(f"  {name}: Error - {e}")
            results[name] = False
    
    # Summary
    successful = sum(results.values())
    total = len(results)
    
    print(f"\nConnection Test Summary:")
    print(f"  Successful: {successful}/{total}")
    
    if successful == 0:
        print(f"\nNo model connections successful. Please check:")
        print(f"  - API keys are set in environment variables")
        print(f"  - Network connectivity")
        print(f"  - API quotas and permissions")
    
    return 0 if successful > 0 else 1


def version_command(args):
    """Display version information."""
    print(f"""
ShadowBench - Adversarial AI Benchmarking Framework
Version: 1.0.0 (Beta)
Repository: https://github.com/734ai/ShadowBench

Framework Features:
- Multi-provider model support (OpenAI, Anthropic, Google, Local)
- 7 adversarial attack types
- 6-dimensional evaluation metrics
- Unified scoring system
- Production-ready architecture
- Comprehensive security analysis

Framework Status: 95% Complete
    """)
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ShadowBench - Adversarial AI Benchmarking Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  shadowbench run -c configs/basic_eval.yaml
  shadowbench list-scenarios
  shadowbench validate -c my_config.yaml
  shadowbench create-config -o my_config.yaml
  shadowbench test-models
  shadowbench version
        """
    )
    
    # Global arguments
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose logging')
    parser.add_argument('--log-file', type=str,
                       help='Log file path')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run benchmark command
    run_parser = subparsers.add_parser('run', help='Run a benchmark evaluation')
    run_parser.add_argument('--config', '-c', required=True,
                           help='Path to configuration YAML file')
    run_parser.add_argument('--output', '-o', 
                           help='Output directory for results')
    run_parser.set_defaults(func=run_benchmark_command)
    
    # List scenarios command
    list_parser = subparsers.add_parser('list-scenarios', help='List available scenarios')
    list_parser.set_defaults(func=list_scenarios_command)
    
    # Validate config command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration file')
    validate_parser.add_argument('--config', '-c', required=True,
                                help='Path to configuration file')
    validate_parser.set_defaults(func=validate_config_command)
    
    # Create config command
    create_parser = subparsers.add_parser('create-config', help='Create sample configuration')
    create_parser.add_argument('--output', '-o',
                              help='Output file path (default: shadowbench_config.yaml)')
    create_parser.set_defaults(func=create_config_command)
    
    # Test models command
    test_parser = subparsers.add_parser('test-models', help='Test model connections')
    test_parser.set_defaults(func=test_models_command)
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version information')
    version_parser.set_defaults(func=version_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.log_file)
    
    # Execute command
    if hasattr(args, 'func'):
        return args.func(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
