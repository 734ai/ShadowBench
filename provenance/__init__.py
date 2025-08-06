"""
Provenance and Chain-of-Trust Module for ShadowBench
Provides comprehensive provenance tracking, cryptographic verification, and audit trails.
"""

from .provenance_tracker import (
    ProvenanceTracker,
    ProvenanceEvent,
    ProvenanceEventType,
    IntegrityCheckResult,
    CryptographicVerifier
)

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import json
from pathlib import Path


class ShadowBenchProvenanceManager:
    """
    High-level provenance management for ShadowBench operations.
    
    This manager integrates with the core benchmark framework to automatically
    track provenance events and maintain chain-of-trust for all operations.
    """
    
    def __init__(self, config: Optional[Dict] = None, storage_path: Optional[str] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize core provenance tracker
        self.tracker = ProvenanceTracker(self.config)
        
        # Storage configuration
        self.storage_path = Path(storage_path) if storage_path else Path("./provenance_data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Auto-save configuration
        self.auto_save_enabled = self.config.get('auto_save_enabled', True)
        self.auto_save_interval = self.config.get('auto_save_interval', 50)  # Events
        self._events_since_save = 0
        
        self.logger.info("ShadowBench Provenance Manager initialized")
    
    def start_benchmark_session(self, benchmark_id: str, 
                               configuration: Dict[str, Any],
                               user_context: Optional[Dict] = None) -> str:
        """
        Start a new benchmark session with provenance tracking.
        
        Args:
            benchmark_id: Unique identifier for the benchmark
            configuration: Benchmark configuration
            user_context: Additional user context
            
        Returns:
            Event ID for the benchmark session start
        """
        metadata = {
            "benchmark_id": benchmark_id,
            "configuration": configuration,
            "user_context": user_context or {},
            "session_start_time": datetime.now(timezone.utc).isoformat(),
            "framework_version": "1.0.0"  # Could be dynamic
        }
        
        event_id = self.tracker.record_event(
            event_type=ProvenanceEventType.BENCHMARK_CREATED,
            actor="shadowbench_framework",
            resource=benchmark_id,
            action="initialize_benchmark_session",
            input_data=configuration,
            metadata=metadata
        )
        
        self._handle_auto_save()
        self.logger.info(f"Started benchmark session {benchmark_id} with provenance event {event_id}")
        
        return event_id
    
    def record_scenario_execution(self, scenario_id: str, scenario_data: Dict[str, Any],
                                 parent_event_id: Optional[str] = None) -> str:
        """
        Record scenario loading and execution.
        
        Args:
            scenario_id: Unique scenario identifier
            scenario_data: Scenario configuration and data
            parent_event_id: Parent benchmark session event
            
        Returns:
            Event ID for scenario execution
        """
        event_id = self.tracker.record_event(
            event_type=ProvenanceEventType.SCENARIO_LOADED,
            actor="shadowbench_framework",
            resource=scenario_id,
            action="load_and_execute_scenario",
            input_data=scenario_data,
            metadata={
                "scenario_type": scenario_data.get("type", "unknown"),
                "scenario_version": scenario_data.get("version", "1.0"),
                "execution_timestamp": datetime.now(timezone.utc).isoformat()
            },
            parent_event_id=parent_event_id
        )
        
        self._handle_auto_save()
        return event_id
    
    def record_model_interaction(self, model_id: str, query_data: Any,
                                response_data: Any, model_metadata: Optional[Dict] = None,
                                parent_event_id: Optional[str] = None) -> str:
        """
        Record model query and response.
        
        Args:
            model_id: Model identifier
            query_data: Query sent to model
            response_data: Response from model
            model_metadata: Additional model information
            parent_event_id: Parent scenario event
            
        Returns:
            Event ID for model interaction
        """
        metadata = {
            "model_id": model_id,
            "query_timestamp": datetime.now(timezone.utc).isoformat(),
            "model_metadata": model_metadata or {}
        }
        
        event_id = self.tracker.record_event(
            event_type=ProvenanceEventType.MODEL_QUERIED,
            actor=f"model_{model_id}",
            resource=f"model_interaction_{model_id}",
            action="query_and_response",
            input_data=query_data,
            output_data=response_data,
            metadata=metadata,
            parent_event_id=parent_event_id
        )
        
        self._handle_auto_save()
        return event_id
    
    def record_attack_execution(self, attack_type: str, attack_config: Dict[str, Any],
                               target_data: Any, attack_result: Any,
                               parent_event_id: Optional[str] = None) -> str:
        """
        Record adversarial attack execution.
        
        Args:
            attack_type: Type of attack (e.g., 'image_adversarial', 'prompt_injection')
            attack_config: Attack configuration
            target_data: Data being attacked
            attack_result: Result of the attack
            parent_event_id: Parent event ID
            
        Returns:
            Event ID for attack execution
        """
        metadata = {
            "attack_type": attack_type,
            "attack_config": attack_config,
            "attack_timestamp": datetime.now(timezone.utc).isoformat(),
            "success_rate": attack_result.get("success_rate", 0.0) if isinstance(attack_result, dict) else None
        }
        
        event_id = self.tracker.record_event(
            event_type=ProvenanceEventType.ATTACK_EXECUTED,
            actor="adversarial_system",
            resource=f"attack_{attack_type}",
            action="execute_adversarial_attack",
            input_data=target_data,
            output_data=attack_result,
            metadata=metadata,
            parent_event_id=parent_event_id
        )
        
        self._handle_auto_save()
        return event_id
    
    def record_metric_calculation(self, metric_name: str, input_data: Any,
                                 metric_value: float, metric_details: Optional[Dict] = None,
                                 parent_event_id: Optional[str] = None) -> str:
        """
        Record metric calculation.
        
        Args:
            metric_name: Name of the metric
            input_data: Data used for metric calculation
            metric_value: Calculated metric value
            metric_details: Additional metric details
            parent_event_id: Parent event ID
            
        Returns:
            Event ID for metric calculation
        """
        output_data = {
            "metric_value": metric_value,
            "metric_details": metric_details or {}
        }
        
        metadata = {
            "metric_name": metric_name,
            "calculation_timestamp": datetime.now(timezone.utc).isoformat(),
            "metric_type": metric_details.get("type", "unknown") if metric_details else "unknown"
        }
        
        event_id = self.tracker.record_event(
            event_type=ProvenanceEventType.METRIC_CALCULATED,
            actor="metrics_system",
            resource=f"metric_{metric_name}",
            action="calculate_metric",
            input_data=input_data,
            output_data=output_data,
            metadata=metadata,
            parent_event_id=parent_event_id
        )
        
        self._handle_auto_save()
        return event_id
    
    def record_result_generation(self, result_type: str, result_data: Dict[str, Any],
                                source_events: Optional[List[str]] = None) -> str:
        """
        Record final result generation.
        
        Args:
            result_type: Type of result (e.g., 'benchmark_report', 'analysis_summary')
            result_data: Generated result data
            source_events: List of source event IDs that contributed to this result
            
        Returns:
            Event ID for result generation
        """
        metadata = {
            "result_type": result_type,
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "source_events": source_events or [],
            "result_size": len(json.dumps(result_data)) if result_data else 0
        }
        
        event_id = self.tracker.record_event(
            event_type=ProvenanceEventType.RESULT_GENERATED,
            actor="result_generator",
            resource=f"result_{result_type}",
            action="generate_final_result",
            output_data=result_data,
            metadata=metadata
        )
        
        self._handle_auto_save()
        return event_id
    
    def validate_benchmark_integrity(self, benchmark_id: str) -> Dict[str, Any]:
        """
        Validate integrity of entire benchmark execution.
        
        Args:
            benchmark_id: Benchmark to validate
            
        Returns:
            Comprehensive integrity report
        """
        # Find all events related to this benchmark
        benchmark_events = [
            event for event in self.tracker.events
            if benchmark_id in event.resource or 
               (event.metadata and event.metadata.get("benchmark_id") == benchmark_id)
        ]
        
        if not benchmark_events:
            return {
                "is_valid": False,
                "benchmark_id": benchmark_id,
                "error": "No events found for benchmark",
                "validation_timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        # Find root benchmark event
        root_event = None
        for event in benchmark_events:
            if (event.event_type == ProvenanceEventType.BENCHMARK_CREATED and 
                event.resource == benchmark_id):
                root_event = event
                break
        
        if not root_event:
            return {
                "is_valid": False,
                "benchmark_id": benchmark_id,
                "error": "No root benchmark event found",
                "validation_timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        # Verify chain integrity from root event
        chain_result = self.tracker.verify_chain_integrity(root_event.event_id)
        
        # Verify individual events
        event_validations = {}
        for event in benchmark_events:
            validation = self.tracker.verify_event_integrity(event.event_id)
            event_validations[event.event_id] = {
                "is_valid": validation.is_valid,
                "confidence_score": validation.confidence_score,
                "issues": validation.issues_found
            }
        
        # Calculate overall integrity score
        valid_events = sum(1 for v in event_validations.values() if v["is_valid"])
        integrity_score = valid_events / len(event_validations) if event_validations else 0.0
        
        # Generate comprehensive report
        integrity_report = {
            "is_valid": chain_result.is_valid and integrity_score > 0.8,
            "benchmark_id": benchmark_id,
            "validation_timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_integrity_score": integrity_score,
            "chain_integrity": {
                "is_valid": chain_result.is_valid,
                "confidence_score": chain_result.confidence_score,
                "chain_length": chain_result.chain_length,
                "trust_score": chain_result.trust_score,
                "issues": chain_result.issues_found
            },
            "event_statistics": {
                "total_events": len(benchmark_events),
                "valid_events": valid_events,
                "invalid_events": len(event_validations) - valid_events,
                "event_types": self._count_event_types(benchmark_events)
            },
            "event_validations": event_validations,
            "recommendations": self._generate_integrity_recommendations(chain_result, event_validations)
        }
        
        return integrity_report
    
    def generate_compliance_report(self, compliance_framework: str = "ISO27001") -> Dict[str, Any]:
        """
        Generate compliance report for audit purposes.
        
        Args:
            compliance_framework: Target compliance framework
            
        Returns:
            Compliance assessment report
        """
        audit_trail = self.tracker.generate_audit_trail()
        
        # Framework-specific compliance checks
        compliance_checks = {
            "ISO27001": self._check_iso27001_compliance,
            "SOX": self._check_sox_compliance,
            "GDPR": self._check_gdpr_compliance
        }
        
        check_function = compliance_checks.get(compliance_framework, self._check_generic_compliance)
        compliance_results = check_function(audit_trail)
        
        compliance_report = {
            "compliance_framework": compliance_framework,
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_compliance_score": compliance_results.get("overall_score", 0.0),
            "audit_trail_summary": {
                "total_events": audit_trail["total_events"],
                "verified_events": audit_trail["integrity_summary"]["verified_events"],
                "average_confidence": audit_trail["integrity_summary"]["average_confidence"]
            },
            "compliance_checks": compliance_results,
            "recommendations": compliance_results.get("recommendations", []),
            "attestation": {
                "generated_by": "ShadowBench Provenance Manager",
                "framework_version": "1.0.0",
                "signature": self.tracker.verifier.compute_hmac(audit_trail)
            }
        }
        
        return compliance_report
    
    def export_provenance_report(self, output_path: str, include_full_data: bool = True) -> bool:
        """
        Export comprehensive provenance report.
        
        Args:
            output_path: Path to save the report
            include_full_data: Include full event data
            
        Returns:
            True if export successful
        """
        try:
            report = {
                "report_metadata": {
                    "generation_timestamp": datetime.now(timezone.utc).isoformat(),
                    "framework": "ShadowBench",
                    "version": "1.0.0",
                    "report_type": "comprehensive_provenance"
                },
                "summary": {
                    "total_events": len(self.tracker.events),
                    "event_types": self._count_event_types(self.tracker.events),
                    "date_range": self._get_event_date_range(),
                    "actors": list(self.tracker.trusted_actors.keys()),
                    "integrity_checkpoints": len(self.tracker.integrity_checkpoints)
                }
            }
            
            if include_full_data:
                # Generate full audit trail
                audit_trail = self.tracker.generate_audit_trail()
                report["full_audit_trail"] = audit_trail
                
                # Add trust scores
                report["trust_scores"] = self.tracker.trusted_actors
                
                # Add integrity checkpoints
                report["integrity_checkpoints"] = self.tracker.integrity_checkpoints
            
            # Export to file
            return self.tracker.export_provenance_data(
                filepath=output_path,
                format="json",
                include_verification=True
            )
            
        except Exception as e:
            self.logger.error(f"Failed to export provenance report: {e}")
            return False
    
    def load_provenance_data(self, filepath: str, verify_integrity: bool = True) -> bool:
        """
        Load provenance data from file.
        
        Args:
            filepath: Path to provenance data file
            verify_integrity: Verify data integrity during load
            
        Returns:
            True if load successful
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Verify signature if present
            if verify_integrity and "signature" in data:
                signature_data = {k: v for k, v in data.items() if k != "signature"}
                expected_signature = self.tracker.verifier.compute_hmac(signature_data)
                
                if data["signature"] != expected_signature:
                    self.logger.warning("Provenance data signature verification failed")
                    return False
            
            # Load events
            if "events" in data:
                for event_data in data["events"]:
                    # Reconstruct ProvenanceEvent object
                    event_data["timestamp"] = datetime.fromisoformat(event_data["timestamp"])
                    event_data["event_type"] = ProvenanceEventType(event_data["event_type"])
                    
                    event = ProvenanceEvent(**event_data)
                    self.tracker.events.append(event)
                    self.tracker.event_index[event.event_id] = event
            
            # Load trust scores
            if "trust_scores" in data:
                self.tracker.trusted_actors.update(data["trust_scores"])
            
            # Load integrity checkpoints
            if "integrity_checkpoints" in data:
                self.tracker.integrity_checkpoints.extend(data["integrity_checkpoints"])
            
            self.logger.info(f"Loaded provenance data from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load provenance data: {e}")
            return False
    
    def _handle_auto_save(self):
        """Handle automatic saving of provenance data."""
        self._events_since_save += 1
        
        if (self.auto_save_enabled and 
            self._events_since_save >= self.auto_save_interval):
            
            auto_save_path = self.storage_path / f"provenance_auto_{int(datetime.now().timestamp())}.json"
            self.export_provenance_report(str(auto_save_path), include_full_data=False)
            self._events_since_save = 0
    
    def _count_event_types(self, events: List[ProvenanceEvent]) -> Dict[str, int]:
        """Count events by type."""
        counts = {}
        for event in events:
            event_type = event.event_type.value
            counts[event_type] = counts.get(event_type, 0) + 1
        return counts
    
    def _get_event_date_range(self) -> Dict[str, str]:
        """Get date range of events."""
        if not self.tracker.events:
            return {"start": None, "end": None}
        
        timestamps = [event.timestamp for event in self.tracker.events]
        return {
            "start": min(timestamps).isoformat(),
            "end": max(timestamps).isoformat()
        }
    
    def _generate_integrity_recommendations(self, chain_result: IntegrityCheckResult,
                                          event_validations: Dict[str, Dict]) -> List[str]:
        """Generate recommendations for improving integrity."""
        recommendations = []
        
        if chain_result.confidence_score < 0.8:
            recommendations.append("Consider enabling digital signatures for stronger integrity guarantees")
        
        if chain_result.trust_score < 0.7:
            recommendations.append("Review actor trust scores and consider additional validation")
        
        invalid_events = sum(1 for v in event_validations.values() if not v["is_valid"])
        if invalid_events > 0:
            recommendations.append(f"Investigate {invalid_events} events with integrity issues")
        
        if len(chain_result.issues_found) > 0:
            recommendations.append("Address chain integrity issues: " + ", ".join(chain_result.issues_found))
        
        return recommendations
    
    def _check_iso27001_compliance(self, audit_trail: Dict[str, Any]) -> Dict[str, Any]:
        """Check ISO 27001 compliance requirements."""
        return {
            "overall_score": 0.85,  # Simplified scoring
            "audit_trail_completeness": audit_trail["total_events"] > 0,
            "integrity_verification": audit_trail["integrity_summary"]["average_confidence"] > 0.7,
            "access_logging": True,  # Based on event tracking
            "recommendations": [
                "Ensure regular integrity checkpoints",
                "Implement role-based access controls"
            ]
        }
    
    def _check_sox_compliance(self, audit_trail: Dict[str, Any]) -> Dict[str, Any]:
        """Check SOX compliance requirements."""
        return {
            "overall_score": 0.80,
            "financial_controls": False,  # Not applicable for AI benchmarking
            "data_integrity": audit_trail["integrity_summary"]["average_confidence"] > 0.8,
            "audit_trail_retention": True,
            "recommendations": [
                "Implement financial impact tracking if applicable",
                "Ensure audit trail immutability"
            ]
        }
    
    def _check_gdpr_compliance(self, audit_trail: Dict[str, Any]) -> Dict[str, Any]:
        """Check GDPR compliance requirements."""
        return {
            "overall_score": 0.75,
            "data_processing_logging": audit_trail["total_events"] > 0,
            "consent_tracking": False,  # Would need specific implementation
            "data_retention": True,
            "recommendations": [
                "Implement consent tracking mechanisms",
                "Add data subject rights handling",
                "Ensure data minimization principles"
            ]
        }
    
    def _check_generic_compliance(self, audit_trail: Dict[str, Any]) -> Dict[str, Any]:
        """Generic compliance check."""
        return {
            "overall_score": 0.70,
            "audit_trail_present": audit_trail["total_events"] > 0,
            "integrity_checks": audit_trail["integrity_summary"]["average_confidence"] > 0.6,
            "recommendations": [
                "Specify compliance framework for detailed assessment"
            ]
        }


# Factory function for easy initialization
def create_provenance_manager(config: Optional[Dict] = None, 
                            storage_path: Optional[str] = None) -> ShadowBenchProvenanceManager:
    """
    Factory function to create a configured provenance manager.
    
    Args:
        config: Configuration dictionary
        storage_path: Path for provenance data storage
        
    Returns:
        Configured ShadowBenchProvenanceManager instance
    """
    return ShadowBenchProvenanceManager(config=config, storage_path=storage_path)


# Context manager for automatic provenance tracking
class ProvenanceContext:
    """Context manager for automatic provenance tracking of operations."""
    
    def __init__(self, manager: ShadowBenchProvenanceManager, 
                 operation_type: str, operation_id: str,
                 parent_event_id: Optional[str] = None):
        self.manager = manager
        self.operation_type = operation_type
        self.operation_id = operation_id
        self.parent_event_id = parent_event_id
        self.start_event_id = None
    
    def __enter__(self):
        # Record operation start
        if self.operation_type == "benchmark":
            self.start_event_id = self.manager.start_benchmark_session(
                self.operation_id, {}, {}
            )
        return self.start_event_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Record operation completion or failure
        if exc_type is None:
            # Successful completion
            self.manager.tracker.record_event(
                event_type=ProvenanceEventType.VALIDATION_PERFORMED,
                actor="context_manager",
                resource=self.operation_id,
                action="operation_completed_successfully",
                parent_event_id=self.start_event_id
            )
        else:
            # Failed completion
            self.manager.tracker.record_event(
                event_type=ProvenanceEventType.VALIDATION_PERFORMED,
                actor="context_manager",
                resource=self.operation_id,
                action="operation_failed",
                metadata={"error": str(exc_val)},
                parent_event_id=self.start_event_id
            )


__all__ = [
    'ProvenanceTracker',
    'ProvenanceEvent', 
    'ProvenanceEventType',
    'IntegrityCheckResult',
    'CryptographicVerifier',
    'ShadowBenchProvenanceManager',
    'ProvenanceContext',
    'create_provenance_manager'
]
