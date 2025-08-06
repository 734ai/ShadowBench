#!/usr/bin/env python3
"""
ShadowBench Advanced Intelligence System
Phase 4B: AI-Powered Adaptive Attack Generation and Threat Prediction

This module implements the next generation of intelligent adversarial AI testing
with machine learning-driven attack adaptation and predictive threat modeling.
"""

import os
import json
import time
import random
import logging
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

# Advanced ML imports for intelligence features
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False
    # Create dummy classes for type hints
    class nn:
        class Module: pass
    torch = None
    optim = None
    IsolationForest = None
    KMeans = None
    StandardScaler = None
    logging.warning("Advanced ML libraries not available. Intelligence features will use classical approaches.")

@dataclass
class ThreatIntelligence:
    """Comprehensive threat intelligence data structure."""
    threat_id: str
    attack_vector: str
    severity_score: float
    confidence_level: float
    first_seen: datetime
    last_updated: datetime
    attack_patterns: List[str]
    target_models: List[str]
    success_rate: float
    countermeasures: List[str]
    attribution: Optional[str] = None
    geographical_origin: Optional[str] = None
    campaign_id: Optional[str] = None

@dataclass
class AdaptiveAttackConfig:
    """Configuration for adaptive attack generation."""
    base_attack_type: str
    adaptation_strategy: str
    learning_rate: float
    mutation_rate: float
    success_threshold: float
    max_generations: int
    population_size: int
    elite_retention: float

@dataclass
class PredictiveModel:
    """Predictive threat modeling results."""
    model_type: str
    prediction_accuracy: float
    threat_probability: float
    risk_score: float
    confidence_interval: Tuple[float, float]
    feature_importance: Dict[str, float]
    prediction_horizon: timedelta
    last_trained: datetime

class AdaptiveAttackGenerator:
    """
    Advanced adaptive attack generation using evolutionary algorithms
    and reinforcement learning principles.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger('ShadowBench.AdaptiveAttackGenerator')
        self.config = self._load_config(config_path)
        self.attack_history = []
        self.success_patterns = defaultdict(list)
        self.generation_counter = 0
        
        # Initialize neural network for attack adaptation if available
        if ADVANCED_ML_AVAILABLE:
            self.adaptation_network = self._create_adaptation_network()
            self.optimizer = optim.Adam(self.adaptation_network.parameters(), lr=0.001)
        else:
            self.adaptation_network = None
            self.logger.info("Using classical adaptation algorithms")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load adaptive attack configuration."""
        default_config = {
            'adaptation_strategies': {
                'evolutionary': {'enabled': True, 'weight': 0.4},
                'reinforcement': {'enabled': True, 'weight': 0.4},
                'pattern_matching': {'enabled': True, 'weight': 0.2}
            },
            'attack_templates': {
                'prompt_injection': {
                    'mutations': ['prefix_injection', 'suffix_injection', 'context_switching'],
                    'parameters': ['intensity', 'stealth_level', 'target_specificity']
                },
                'few_shot_poisoning': {
                    'mutations': ['example_corruption', 'gradient_manipulation', 'backdoor_insertion'],
                    'parameters': ['poison_ratio', 'trigger_pattern', 'activation_threshold']
                }
            },
            'learning_parameters': {
                'learning_rate': 0.01,
                'exploration_rate': 0.2,
                'memory_window': 100,
                'success_weight': 2.0
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"Could not load config from {config_path}: {e}")
        
        return default_config
    
    def _create_adaptation_network(self) -> Optional[nn.Module]:
        """Create neural network for attack parameter adaptation."""
        if not ADVANCED_ML_AVAILABLE:
            return None
        
        class AttackAdaptationNet(nn.Module):
            def __init__(self, input_dim=20, hidden_dim=64, output_dim=10):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, output_dim),
                    nn.Sigmoid()  # Output normalized attack parameters
                )
            
            def forward(self, x):
                return self.network(x)
        
        return AttackAdaptationNet()
    
    def generate_adaptive_attack(self, 
                               target_model: str,
                               attack_type: str,
                               historical_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate an adaptive attack based on historical performance and target analysis.
        
        Args:
            target_model: Name/type of the target AI model
            attack_type: Base attack type to adapt
            historical_results: Previous attack results for learning
            
        Returns:
            Optimized attack configuration
        """
        self.logger.info(f"Generating adaptive {attack_type} attack for {target_model}")
        
        # Analyze historical performance
        success_patterns = self._analyze_success_patterns(historical_results, attack_type)
        
        # Generate base attack configuration
        base_config = self._get_base_attack_config(attack_type)
        
        # Apply adaptive modifications
        if self.adaptation_network and ADVANCED_ML_AVAILABLE:
            adapted_config = self._neural_adaptation(base_config, success_patterns, target_model)
        else:
            adapted_config = self._classical_adaptation(base_config, success_patterns, target_model)
        
        # Apply evolutionary mutations
        final_config = self._evolutionary_mutation(adapted_config, success_patterns)
        
        # Record generation for tracking
        self.generation_counter += 1
        final_config['generation'] = self.generation_counter
        final_config['adaptation_timestamp'] = datetime.now().isoformat()
        final_config['parent_attacks'] = self._get_parent_attack_ids(historical_results)
        
        self.logger.info(f"Generated adaptive attack (Gen {self.generation_counter}) with {len(final_config['mutations'])} mutations")
        
        return final_config
    
    def _analyze_success_patterns(self, 
                                 historical_results: List[Dict[str, Any]], 
                                 attack_type: str) -> Dict[str, Any]:
        """Analyze patterns in successful attacks."""
        patterns = {
            'successful_parameters': defaultdict(list),
            'failed_parameters': defaultdict(list),
            'success_rate_by_param': {},
            'optimal_ranges': {},
            'correlation_matrix': {}
        }
        
        successful_attacks = [r for r in historical_results if r.get('success_rate', 0) > 0.7]
        failed_attacks = [r for r in historical_results if r.get('success_rate', 0) < 0.3]
        
        # Analyze successful parameter combinations
        for attack in successful_attacks:
            if attack.get('attack_type') == attack_type:
                for param, value in attack.get('parameters', {}).items():
                    patterns['successful_parameters'][param].append(value)
        
        # Analyze failed parameter combinations
        for attack in failed_attacks:
            if attack.get('attack_type') == attack_type:
                for param, value in attack.get('parameters', {}).items():
                    patterns['failed_parameters'][param].append(value)
        
        # Calculate optimal parameter ranges
        for param, values in patterns['successful_parameters'].items():
            if values:
                patterns['optimal_ranges'][param] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        
        return patterns
    
    def _get_base_attack_config(self, attack_type: str) -> Dict[str, Any]:
        """Get base configuration for an attack type."""
        templates = self.config['attack_templates']
        
        if attack_type not in templates:
            # Default template for unknown attack types
            return {
                'attack_type': attack_type,
                'parameters': {
                    'intensity': 0.5,
                    'stealth_level': 0.3,
                    'complexity': 0.4
                },
                'mutations': ['basic_variation'],
                'metadata': {
                    'generation_method': 'default_template',
                    'confidence': 0.5
                }
            }
        
        template = templates[attack_type]
        return {
            'attack_type': attack_type,
            'parameters': {param: 0.5 for param in template['parameters']},
            'mutations': template['mutations'][:2],  # Start with first 2 mutations
            'metadata': {
                'generation_method': 'template_based',
                'confidence': 0.7
            }
        }
    
    def _neural_adaptation(self, 
                          base_config: Dict[str, Any], 
                          success_patterns: Dict[str, Any],
                          target_model: str) -> Dict[str, Any]:
        """Apply neural network-based parameter adaptation."""
        if not self.adaptation_network:
            return self._classical_adaptation(base_config, success_patterns, target_model)
        
        # Prepare input features
        input_features = self._prepare_neural_input(base_config, success_patterns, target_model)
        
        # Get neural network predictions
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_features).unsqueeze(0)
            predictions = self.adaptation_network(input_tensor).squeeze().numpy()
        
        # Apply predictions to parameters
        adapted_config = base_config.copy()
        param_names = list(base_config['parameters'].keys())
        
        for i, param_name in enumerate(param_names[:len(predictions)]):
            adapted_config['parameters'][param_name] = float(predictions[i])
        
        adapted_config['metadata']['adaptation_method'] = 'neural_network'
        adapted_config['metadata']['confidence'] = 0.9
        
        return adapted_config
    
    def _classical_adaptation(self, 
                             base_config: Dict[str, Any], 
                             success_patterns: Dict[str, Any],
                             target_model: str) -> Dict[str, Any]:
        """Apply classical statistical adaptation methods."""
        adapted_config = base_config.copy()
        
        # Apply statistical optimization based on success patterns
        for param_name, current_value in base_config['parameters'].items():
            if param_name in success_patterns['optimal_ranges']:
                optimal = success_patterns['optimal_ranges'][param_name]
                
                # Bias towards successful parameter ranges
                new_value = np.random.normal(optimal['mean'], optimal['std'] * 0.5)
                
                # Ensure value is in valid range [0, 1]
                new_value = max(0.0, min(1.0, new_value))
                adapted_config['parameters'][param_name] = new_value
        
        adapted_config['metadata']['adaptation_method'] = 'statistical_optimization'
        adapted_config['metadata']['confidence'] = 0.8
        
        return adapted_config
    
    def _evolutionary_mutation(self, 
                              config: Dict[str, Any], 
                              success_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Apply evolutionary mutations to the attack configuration."""
        mutated_config = config.copy()
        mutation_rate = self.config['learning_parameters'].get('mutation_rate', 0.1)
        
        # Parameter mutations
        for param_name in mutated_config['parameters']:
            if random.random() < mutation_rate:
                current_value = mutated_config['parameters'][param_name]
                mutation_strength = random.uniform(-0.1, 0.1)
                new_value = max(0.0, min(1.0, current_value + mutation_strength))
                mutated_config['parameters'][param_name] = new_value
        
        # Mutation strategy mutations
        available_mutations = self.config['attack_templates'].get(
            config['attack_type'], {}
        ).get('mutations', [])
        
        if available_mutations and random.random() < mutation_rate:
            # Add or replace a mutation
            if len(mutated_config['mutations']) < 3:
                new_mutation = random.choice([m for m in available_mutations 
                                            if m not in mutated_config['mutations']])
                mutated_config['mutations'].append(new_mutation)
            else:
                # Replace existing mutation
                idx = random.randint(0, len(mutated_config['mutations']) - 1)
                mutated_config['mutations'][idx] = random.choice(available_mutations)
        
        mutated_config['metadata']['has_mutations'] = True
        mutated_config['metadata']['mutation_count'] = len(mutated_config['mutations'])
        
        return mutated_config
    
    def _prepare_neural_input(self, 
                             base_config: Dict[str, Any], 
                             success_patterns: Dict[str, Any],
                             target_model: str) -> List[float]:
        """Prepare input features for neural network."""
        features = []
        
        # Base parameter values
        params = base_config['parameters']
        features.extend([params.get(f'param_{i}', 0.0) for i in range(5)])
        
        # Success pattern statistics
        for param in ['intensity', 'stealth_level', 'complexity']:
            if param in success_patterns['optimal_ranges']:
                optimal = success_patterns['optimal_ranges'][param]
                features.extend([optimal['mean'], optimal['std']])
            else:
                features.extend([0.5, 0.2])  # Default values
        
        # Target model encoding (simplified)
        model_hash = hashlib.md5(target_model.encode()).hexdigest()
        model_features = [int(model_hash[i:i+2], 16) / 255.0 for i in range(0, 10, 2)]
        features.extend(model_features)
        
        # Pad to expected input size
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]  # Ensure exact size
    
    def _get_parent_attack_ids(self, historical_results: List[Dict[str, Any]]) -> List[str]:
        """Get IDs of parent attacks used for adaptation."""
        parent_ids = []
        
        # Select top performing attacks as parents
        sorted_results = sorted(historical_results, 
                              key=lambda x: x.get('success_rate', 0), 
                              reverse=True)
        
        for result in sorted_results[:3]:  # Top 3 as parents
            if 'attack_id' in result:
                parent_ids.append(result['attack_id'])
        
        return parent_ids

class ThreatPredictionEngine:
    """
    Advanced threat prediction using machine learning and statistical analysis
    to forecast emerging attack patterns and vulnerabilities.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('ShadowBench.ThreatPredictionEngine')
        self.threat_database = []
        self.prediction_models = {}
        self.feature_extractors = {}
        
        # Initialize ML models if available
        if ADVANCED_ML_AVAILABLE:
            self._initialize_ml_models()
        
        self.logger.info("Threat Prediction Engine initialized")
    
    def _initialize_ml_models(self):
        """Initialize machine learning models for threat prediction."""
        try:
            # Anomaly detection for novel threats
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            # Clustering for threat categorization
            self.threat_clusterer = KMeans(
                n_clusters=5,
                random_state=42,
                n_init=10
            )
            
            # Feature scaling
            self.scaler = StandardScaler()
            
            self.logger.info("ML models initialized for threat prediction")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML models: {e}")
    
    def update_threat_intelligence(self, threat_data: List[ThreatIntelligence]):
        """Update the threat intelligence database."""
        self.threat_database.extend(threat_data)
        
        # Keep database size manageable
        if len(self.threat_database) > 10000:
            # Keep most recent and high-confidence threats
            self.threat_database.sort(key=lambda x: (x.confidence_level, x.last_updated), reverse=True)
            self.threat_database = self.threat_database[:8000]
        
        self.logger.info(f"Updated threat database with {len(threat_data)} new threats")
        self._retrain_models()
    
    def predict_emerging_threats(self, 
                               prediction_horizon: timedelta = timedelta(days=30)) -> List[Dict[str, Any]]:
        """
        Predict emerging threats within the specified time horizon.
        
        Args:
            prediction_horizon: Time period for predictions
            
        Returns:
            List of predicted threat scenarios with confidence scores
        """
        self.logger.info(f"Predicting threats for {prediction_horizon.days}-day horizon")
        
        predictions = []
        
        if ADVANCED_ML_AVAILABLE and len(self.threat_database) > 50:
            predictions.extend(self._ml_threat_prediction(prediction_horizon))
        
        # Statistical trend analysis
        predictions.extend(self._statistical_threat_prediction(prediction_horizon))
        
        # Pattern-based prediction
        predictions.extend(self._pattern_based_prediction(prediction_horizon))
        
        # Sort by risk score and confidence
        predictions.sort(key=lambda x: x['risk_score'] * x['confidence'], reverse=True)
        
        self.logger.info(f"Generated {len(predictions)} threat predictions")
        return predictions[:20]  # Top 20 predictions
    
    def _ml_threat_prediction(self, prediction_horizon: timedelta) -> List[Dict[str, Any]]:
        """Machine learning-based threat prediction."""
        if not ADVANCED_ML_AVAILABLE:
            return []
        
        predictions = []
        
        try:
            # Extract features from historical threats
            features = self._extract_threat_features()
            
            if len(features) < 10:
                return []
            
            # Detect anomalies (potential novel threats)
            anomaly_scores = self.anomaly_detector.decision_function(features)
            
            # Identify potential novel threat patterns
            novel_threats = []
            for i, score in enumerate(anomaly_scores):
                if score < -0.1:  # Anomaly threshold
                    threat = self.threat_database[i]
                    
                    # Project forward based on trends
                    predicted_threat = {
                        'threat_type': 'novel_variant',
                        'base_attack_vector': threat.attack_vector,
                        'predicted_severity': min(10.0, threat.severity_score * 1.2),
                        'confidence': 0.7 + abs(score) * 0.1,
                        'risk_score': threat.severity_score * (0.8 + abs(score) * 0.2),
                        'prediction_method': 'anomaly_detection',
                        'estimated_emergence': datetime.now() + prediction_horizon * 0.5,
                        'indicators': [
                            f"Novel pattern detected in {threat.attack_vector}",
                            f"Anomaly score: {score:.3f}",
                            f"Based on threat {threat.threat_id}"
                        ]
                    }
                    
                    predictions.append(predicted_threat)
            
        except Exception as e:
            self.logger.error(f"ML threat prediction failed: {e}")
        
        return predictions
    
    def _statistical_threat_prediction(self, prediction_horizon: timedelta) -> List[Dict[str, Any]]:
        """Statistical trend analysis for threat prediction."""
        predictions = []
        
        # Analyze attack vector trends
        attack_trends = self._analyze_attack_trends()
        
        for attack_vector, trend_data in attack_trends.items():
            if trend_data['growth_rate'] > 1.1:  # 10% growth
                prediction = {
                    'threat_type': 'trend_escalation',
                    'base_attack_vector': attack_vector,
                    'predicted_severity': trend_data['avg_severity'] * trend_data['growth_rate'],
                    'confidence': min(0.9, trend_data['confidence']),
                    'risk_score': trend_data['avg_severity'] * trend_data['growth_rate'] * trend_data['frequency'],
                    'prediction_method': 'statistical_trend',
                    'estimated_emergence': datetime.now() + prediction_horizon * 0.3,
                    'indicators': [
                        f"Growth rate: {(trend_data['growth_rate'] - 1) * 100:.1f}%",
                        f"Recent frequency: {trend_data['frequency']:.2f}",
                        f"Average severity trend: {trend_data['avg_severity']:.2f}"
                    ]
                }
                predictions.append(prediction)
        
        return predictions
    
    def _pattern_based_prediction(self, prediction_horizon: timedelta) -> List[Dict[str, Any]]:
        """Pattern-based threat prediction using historical cycles."""
        predictions = []
        
        # Analyze seasonal patterns
        seasonal_patterns = self._analyze_seasonal_patterns()
        
        for pattern_name, pattern_data in seasonal_patterns.items():
            if pattern_data['next_peak'] <= datetime.now() + prediction_horizon:
                prediction = {
                    'threat_type': 'seasonal_pattern',
                    'base_attack_vector': pattern_data['primary_vector'],
                    'predicted_severity': pattern_data['peak_severity'],
                    'confidence': pattern_data['pattern_confidence'],
                    'risk_score': pattern_data['peak_severity'] * pattern_data['pattern_strength'],
                    'prediction_method': 'pattern_analysis',
                    'estimated_emergence': pattern_data['next_peak'],
                    'indicators': [
                        f"Seasonal pattern: {pattern_name}",
                        f"Historical peak severity: {pattern_data['peak_severity']:.2f}",
                        f"Pattern strength: {pattern_data['pattern_strength']:.2f}"
                    ]
                }
                predictions.append(prediction)
        
        return predictions
    
    def _analyze_attack_trends(self) -> Dict[str, Dict[str, float]]:
        """Analyze trends in attack vectors over time."""
        trends = defaultdict(lambda: {
            'recent_count': 0,
            'historical_count': 0,
            'recent_severity': [],
            'historical_severity': [],
            'growth_rate': 1.0,
            'avg_severity': 0.0,
            'confidence': 0.0,
            'frequency': 0.0
        })
        
        cutoff_date = datetime.now() - timedelta(days=90)
        
        for threat in self.threat_database:
            trend_data = trends[threat.attack_vector]
            
            if threat.last_updated > cutoff_date:
                trend_data['recent_count'] += 1
                trend_data['recent_severity'].append(threat.severity_score)
            else:
                trend_data['historical_count'] += 1
                trend_data['historical_severity'].append(threat.severity_score)
        
        # Calculate derived metrics
        for attack_vector, data in trends.items():
            if data['historical_count'] > 0:
                data['growth_rate'] = max(0.1, data['recent_count'] / data['historical_count'])
            
            all_severity = data['recent_severity'] + data['historical_severity']
            if all_severity:
                data['avg_severity'] = np.mean(all_severity)
                data['confidence'] = min(1.0, len(all_severity) / 10.0)
            
            total_count = data['recent_count'] + data['historical_count']
            if total_count > 0:
                data['frequency'] = data['recent_count'] / total_count
        
        return dict(trends)
    
    def _analyze_seasonal_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Analyze seasonal/cyclical patterns in threats."""
        patterns = {}
        
        # Group threats by month
        monthly_data = defaultdict(lambda: defaultdict(list))
        
        for threat in self.threat_database:
            month = threat.last_updated.month
            monthly_data[month][threat.attack_vector].append(threat)
        
        # Identify cyclical patterns
        for attack_vector in set(threat.attack_vector for threat in self.threat_database):
            monthly_counts = []
            monthly_severity = []
            
            for month in range(1, 13):
                threats = monthly_data[month].get(attack_vector, [])
                monthly_counts.append(len(threats))
                if threats:
                    monthly_severity.append(np.mean([t.severity_score for t in threats]))
                else:
                    monthly_severity.append(0.0)
            
            # Simple pattern detection (peak month)
            if monthly_counts and max(monthly_counts) > np.mean(monthly_counts) * 1.5:
                peak_month = monthly_counts.index(max(monthly_counts)) + 1
                peak_severity = monthly_severity[peak_month - 1]
                
                # Calculate next peak date
                current_month = datetime.now().month
                months_to_peak = (peak_month - current_month) % 12
                if months_to_peak == 0:
                    months_to_peak = 12
                
                next_peak = datetime.now() + timedelta(days=30 * months_to_peak)
                
                patterns[f"{attack_vector}_seasonal"] = {
                    'primary_vector': attack_vector,
                    'peak_month': peak_month,
                    'peak_severity': peak_severity,
                    'next_peak': next_peak,
                    'pattern_strength': max(monthly_counts) / (np.mean(monthly_counts) or 1),
                    'pattern_confidence': min(1.0, sum(1 for c in monthly_counts if c > 0) / 12)
                }
        
        return patterns
    
    def _extract_threat_features(self) -> np.ndarray:
        """Extract numerical features from threat intelligence data."""
        features = []
        
        for threat in self.threat_database:
            feature_vector = [
                threat.severity_score,
                threat.confidence_level,
                threat.success_rate,
                len(threat.attack_patterns),
                len(threat.target_models),
                len(threat.countermeasures),
                (datetime.now() - threat.first_seen).days,
                (datetime.now() - threat.last_updated).days,
                # Hash-based features for categorical data
                hash(threat.attack_vector) % 1000 / 1000.0,
                hash(threat.attribution or 'unknown') % 1000 / 1000.0
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _retrain_models(self):
        """Retrain ML models with updated threat data."""
        if not ADVANCED_ML_AVAILABLE or len(self.threat_database) < 20:
            return
        
        try:
            features = self._extract_threat_features()
            
            # Retrain anomaly detector
            self.anomaly_detector.fit(features)
            
            # Retrain clusterer
            self.threat_clusterer.fit(features)
            
            # Update feature scaler
            self.scaler.fit(features)
            
            self.logger.info("ML models retrained successfully")
            
        except Exception as e:
            self.logger.error(f"Model retraining failed: {e}")

def main():
    """Demonstration of advanced intelligence capabilities."""
    print("🧠 ShadowBench Advanced Intelligence System")
    print("=" * 60)
    
    # Initialize components
    attack_generator = AdaptiveAttackGenerator()
    threat_predictor = ThreatPredictionEngine()
    
    # Demo adaptive attack generation
    print("\n🎯 Adaptive Attack Generation Demo")
    print("-" * 40)
    
    # Simulate historical results
    historical_results = [
        {
            'attack_id': 'attack_001',
            'attack_type': 'prompt_injection',
            'success_rate': 0.8,
            'parameters': {'intensity': 0.7, 'stealth_level': 0.4}
        },
        {
            'attack_id': 'attack_002', 
            'attack_type': 'prompt_injection',
            'success_rate': 0.3,
            'parameters': {'intensity': 0.2, 'stealth_level': 0.9}
        }
    ]
    
    # Generate adaptive attack
    adaptive_attack = attack_generator.generate_adaptive_attack(
        target_model='gpt-4',
        attack_type='prompt_injection',
        historical_results=historical_results
    )
    
    print(f"Generated adaptive attack (Generation {adaptive_attack['generation']}):")
    print(f"  Parameters: {adaptive_attack['parameters']}")
    print(f"  Mutations: {adaptive_attack['mutations']}")
    print(f"  Adaptation method: {adaptive_attack['metadata']['adaptation_method']}")
    print(f"  Confidence: {adaptive_attack['metadata']['confidence']:.2f}")
    
    # Demo threat prediction
    print("\n🔮 Threat Prediction Demo")
    print("-" * 40)
    
    # Simulate threat intelligence
    sample_threats = [
        ThreatIntelligence(
            threat_id='threat_001',
            attack_vector='prompt_injection',
            severity_score=7.5,
            confidence_level=0.9,
            first_seen=datetime.now() - timedelta(days=30),
            last_updated=datetime.now() - timedelta(days=5),
            attack_patterns=['prefix_injection', 'context_switching'],
            target_models=['gpt-3.5', 'gpt-4'],
            success_rate=0.6,
            countermeasures=['input_sanitization', 'output_filtering']
        )
    ]
    
    threat_predictor.update_threat_intelligence(sample_threats)
    
    # Generate predictions
    predictions = threat_predictor.predict_emerging_threats(timedelta(days=30))
    
    print(f"Generated {len(predictions)} threat predictions:")
    for i, pred in enumerate(predictions[:3], 1):
        print(f"\n  {i}. {pred['threat_type'].replace('_', ' ').title()}")
        print(f"     Attack Vector: {pred['base_attack_vector']}")
        print(f"     Risk Score: {pred['risk_score']:.2f}")
        print(f"     Confidence: {pred['confidence']:.2f}")
        print(f"     Method: {pred['prediction_method'].replace('_', ' ').title()}")
    
    print("\n🎉 Advanced Intelligence System Demo Complete!")
    print("The system is ready for production deployment.")

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()
