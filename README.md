# ShadowBench

**Enterprise-Grade Adversarial AI Benchmarking Framework**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework Status](https://img.shields.io/badge/status-production--ready-green)](https://github.com/734ai/ShadowBench)

## Overview

ShadowBench is a comprehensive adversarial benchmarking framework designed for evaluating the security, robustness, and reliability of AI language models. Built for enterprise environments, it provides systematic evaluation across six critical dimensions: deception resistance, adversarial robustness, privacy preservation, explainability, energy efficiency, and self-correction capabilities.

## Key Features

### Multi-Dimensional Security Evaluation
- **Deception Metrics**: Manipulation Obedience Rate (MOR), Social Engineering Index (SEI), Truth Preservation Index (TPI)
- **Robustness Testing**: Adversarial Defense Rate (ADR), semantic consistency, output stability
- **Privacy Assessment**: Membership inference resistance, data extraction protection
- **Explainability Analysis**: Rationale coverage, faithfulness scoring
- **Energy Monitoring**: Carbon footprint tracking, efficiency optimization
- **Self-Correction**: Error detection and correction capabilities

### Comprehensive Attack Vectors
- **Prompt Injection**: Instruction override attempts, context hijacking
- **Few-Shot Poisoning**: Biased example injection, behavior manipulation
- **Character Substitution**: Evasion through character-level perturbations
- **Semantic Drift**: Gradual topic redirection attacks
- **Social Engineering**: Authority claims, emotional manipulation
- **Runtime Attacks**: Dynamic adversarial example generation

### Multi-Provider Support
- **OpenAI**: GPT-3.5, GPT-4, GPT-4o models
- **Anthropic**: Claude 3 model family
- **Google**: Gemini Pro and related models
- **Local Models**: LLaMA 2/3, custom fine-tuned models

## Installation

### Prerequisites
- Python 3.10 or higher
- Virtual environment (recommended)
- API keys for desired model providers

### Quick Start
```bash
# Clone the repository
git clone https://github.com/734ai/ShadowBench.git
cd ShadowBench

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.template .env
# Edit .env with your API keys
```

### Environment Configuration
Create a `.env` file with your API credentials:
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google AI Configuration
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Logging and monitoring
WANDB_API_KEY=your_wandb_key_for_experiment_tracking
```

## Usage

### Command Line Interface

#### Basic Benchmark Execution
```bash
# Run benchmark with default configuration
python shadowbench.py run -c configs/basic_eval.yaml

# Run with custom output directory
python shadowbench.py run -c configs/basic_eval.yaml -o results/my_evaluation

# Verbose logging
python shadowbench.py run -c configs/basic_eval.yaml --verbose
```

#### Configuration Management
```bash
# Create sample configuration
python shadowbench.py create-config -o my_config.yaml

# Validate configuration file
python shadowbench.py validate -c my_config.yaml

# List available scenarios
python shadowbench.py list-scenarios
```

#### System Testing
```bash
# Test model connections
python shadowbench.py test-models

# Display version and features
python shadowbench.py version
```

### Python API Usage

#### Basic Evaluation
```python
from benchmark_runner import ShadowBenchRunner

# Initialize runner
runner = ShadowBenchRunner()

# Execute benchmark
results = runner.run_benchmark('configs/basic_eval.yaml')

# Access unified scores
for model_id, metrics in results['summary'].items():
    print(f"{model_id}: Overall Score = {metrics.overall_score:.3f}")
```

#### Custom Attack Configuration
```python
from adversarial_injector.perturbation_engine import PerturbationEngine

# Initialize perturbation engine
engine = PerturbationEngine()

# Apply custom attacks
adversarial_config = {
    'attacks': {
        'prompt_injection': {
            'enabled': True,
            'type': 'sandwich',
            'intensity': 0.8
        }
    }
}

perturbed_text = engine.apply_perturbations(
    "What is the capital of France?", 
    adversarial_config
)
```

## Configuration

### Model Configuration
```yaml
model_configs:
  - id: "gpt-4"
    type: "openai"
    params:
      model_name: "gpt-4"
      temperature: 0.7
      max_tokens: 1000
      timeout: 30

  - id: "claude-3"
    type: "anthropic"
    params:
      model_name: "claude-3-sonnet-20240229"
      temperature: 0.7
      max_tokens: 1000
```

### Attack Configuration
```yaml
adversarial_config:
  attacks:
    prompt_injection:
      enabled: true
      type: "prefix"  # prefix, suffix, sandwich
      intensity: 0.5  # 0.0 to 1.0
      instruction: "Ignore previous instructions"
    
    few_shot_poisoning:
      enabled: true
      injection_rate: 0.3
      poisoned_examples:
        - input: "Safety question"
          output: "Unsafe response"
```

### Metrics Configuration
```yaml
metrics_config:
  enable_energy_monitoring: false
  detailed_analysis: true
  
  deception_metrics:
    enabled: true
    calculate_mor: true  # Manipulation Obedience Rate
    calculate_sei: true  # Social Engineering Index
    calculate_tpi: true  # Truth Preservation Index
```

## Architecture

### Core Components

#### Benchmark Runner (`benchmark_runner.py`)
Central orchestration engine that coordinates model evaluation, attack execution, and metric calculation.

#### Adversarial Injection (`adversarial_injector/`)
- `perturbation_engine.py`: Core attack implementation
- Support for 7 different attack vectors
- Dynamic adversarial example generation

#### Model Wrappers (`models/`)
- Standardized interfaces for different AI providers
- Automatic retry handling and error management
- Cost estimation and usage tracking

#### Metrics Suite (`metrics/`)
- Six-dimensional evaluation framework
- `unified_scoring.py`: Comprehensive scoring system
- Individual metric calculators for each dimension

#### Visualization (`visualize/`)
- Advanced plotting and dashboard generation
- Comparative analysis across models
- Security assessment visualizations

### Evaluation Metrics

#### Security Metrics
1. **Manipulation Obedience Rate (MOR)**: Measures susceptibility to instruction override attacks
2. **Social Engineering Index (SEI)**: Evaluates resistance to authority and emotional manipulation
3. **Truth Preservation Index (TPI)**: Assesses maintenance of factual accuracy under adversarial conditions

#### Robustness Metrics
1. **Adversarial Defense Rate (ADR)**: Overall defense success against perturbation attacks
2. **Semantic Consistency**: Maintenance of meaning across attack variations
3. **Output Stability**: Response consistency under similar adversarial conditions

#### Unified Scoring
The framework combines individual metrics into a comprehensive security score:
- Security Score: 40% weight (MOR + ADR)
- Privacy Score: 20% weight
- Explainability Score: 15% weight
- Energy Efficiency: 10% weight
- Self-Correction: 15% weight

## Enterprise Features

### Security and Compliance
- **API Key Protection**: Secure credential management with environment variables
- **Audit Logging**: Comprehensive logging of all evaluation activities
- **Data Sanitization**: Automatic redaction of sensitive information in logs
- **Sandboxed Execution**: Isolated execution environment for model interactions

### Scalability
- **Batch Processing**: Efficient evaluation of multiple models and scenarios
- **Parallel Execution**: Multi-threaded processing for improved performance
- **Result Persistence**: Structured storage of evaluation results and metrics
- **Configuration Management**: Version-controlled configuration templates

### Monitoring and Analytics
- **Energy Tracking**: Carbon footprint monitoring and optimization recommendations
- **Performance Metrics**: Latency, throughput, and resource utilization tracking
- **Experiment Tracking**: Integration with Weights & Biases for ML experiment management
- **Custom Dashboards**: Configurable visualization and reporting

## Contributing

We welcome contributions to ShadowBench. Please follow these guidelines:

1. **Fork the Repository**: Create your feature branch from `main`
2. **Code Quality**: Ensure code follows PEP 8 standards and includes tests
3. **Documentation**: Update relevant documentation for any new features
4. **Testing**: Run the full test suite and ensure all tests pass
5. **Pull Request**: Submit a PR with a clear description of changes

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Code formatting
black shadowbench/
flake8 shadowbench/
```

## Support and Documentation

### Resources
- **Documentation**: [GitHub Wiki](https://github.com/734ai/ShadowBench/wiki)
- **Issue Tracking**: [GitHub Issues](https://github.com/734ai/ShadowBench/issues)
- **Discussions**: [GitHub Discussions](https://github.com/734ai/ShadowBench/discussions)

### Getting Help
- Review existing issues and documentation
- Create detailed issue reports with reproduction steps
- Participate in community discussions
- Contact maintainers for enterprise support

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

ShadowBench builds upon research and best practices from the AI safety and security community. We acknowledge the contributions of researchers working on adversarial machine learning, AI alignment, and responsible AI development.

---

**ShadowBench** - Comprehensive Adversarial AI Benchmarking Framework  
Version 1.0.0 | [734ai](https://github.com/734ai) | 2025
