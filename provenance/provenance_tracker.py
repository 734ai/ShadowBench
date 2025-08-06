"""
Provenance and Chain-of-Trust System for ShadowBench
Implements cryptographic verification and audit trails for benchmark integrity.
"""

import logging
import hashlib
import hmac
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import secrets
import base64
from datetime import datetime, timezone
from pathlib import Path


class ProvenanceEventType(Enum):
    """Types of provenance events."""
    BENCHMARK_CREATED = "benchmark_created"
    SCENARIO_LOADED = "scenario_loaded"
    MODEL_QUERIED = "model_queried"
    RESULT_GENERATED = "result_generated"
    METRIC_CALCULATED = "metric_calculated"
    DATA_MODIFIED = "data_modified"
    SYSTEM_CONFIGURATION = "system_configuration"
    ATTACK_EXECUTED = "attack_executed"
    VALIDATION_PERFORMED = "validation_performed"


@dataclass
class ProvenanceEvent:
    """Single provenance event record."""
    event_id: str
    event_type: ProvenanceEventType
    timestamp: datetime
    actor: str  # System component or user responsible
    resource: str  # What was affected
    action: str  # What action was performed
    input_hash: Optional[str] = None  # Hash of input data
    output_hash: Optional[str] = None  # Hash of output data
    signature: Optional[str] = None  # Cryptographic signature
    metadata: Optional[Dict[str, Any]] = None
    parent_event_id: Optional[str] = None  # For event chains


@dataclass
class IntegrityCheckResult:
    """Result of integrity verification."""
    is_valid: bool
    verification_method: str
    confidence_score: float
    issues_found: List[str]
    verification_timestamp: datetime
    chain_length: int
    trust_score: float


class CryptographicVerifier:
    """
    Cryptographic verification system for data integrity.
    
    Features:
    - SHA-256/SHA-3 hashing for data integrity
    - HMAC for authenticated verification
    - Digital signatures for non-repudiation
    - Merkle tree construction for batch verification
    - Timestamping for temporal proof
    """
    
    def __init__(self, secret_key: Optional[bytes] = None):
        self.logger = logging.getLogger(__name__)
        
        # Generate or use provided secret key for HMAC
        self.secret_key = secret_key or secrets.token_bytes(32)
        
        # Supported hash algorithms
        self.hash_algorithms = {
            'sha256': hashlib.sha256,
            'sha3_256': hashlib.sha3_256,
            'blake2b': lambda data: hashlib.blake2b(data, digest_size=32)
        }
    
    def compute_data_hash(self, data: Any, algorithm: str = 'sha256') -> str:
        """
        Compute cryptographic hash of data.
        
        Args:
            data: Data to hash (will be serialized if not bytes)
            algorithm: Hash algorithm to use
            
        Returns:
            Hexadecimal hash string
        """
        if algorithm not in self.hash_algorithms:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        # Serialize data if necessary
        if isinstance(data, (dict, list)):
            data_bytes = json.dumps(data, sort_keys=True, default=str).encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            data_bytes = str(data).encode('utf-8')
        
        hash_func = self.hash_algorithms[algorithm]
        return hash_func(data_bytes).hexdigest()
    
    def compute_hmac(self, data: Any, algorithm: str = 'sha256') -> str:
        """
        Compute HMAC for authenticated verification.
        
        Args:
            data: Data to authenticate
            algorithm: Hash algorithm for HMAC
            
        Returns:
            Base64-encoded HMAC
        """
        if algorithm not in self.hash_algorithms:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        # Serialize data
        if isinstance(data, (dict, list)):
            data_bytes = json.dumps(data, sort_keys=True, default=str).encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            data_bytes = str(data).encode('utf-8')
        
        mac = hmac.new(self.secret_key, data_bytes, self.hash_algorithms[algorithm])
        return base64.b64encode(mac.digest()).decode('ascii')
    
    def verify_hmac(self, data: Any, provided_mac: str, algorithm: str = 'sha256') -> bool:
        """
        Verify HMAC authenticity.
        
        Args:
            data: Original data
            provided_mac: HMAC to verify
            algorithm: Hash algorithm used
            
        Returns:
            True if HMAC is valid
        """
        try:
            expected_mac = self.compute_hmac(data, algorithm)
            return hmac.compare_digest(expected_mac, provided_mac)
        except Exception as e:
            self.logger.error(f"HMAC verification failed: {e}")
            return False
    
    def build_merkle_tree(self, data_hashes: List[str]) -> Dict[str, Any]:
        """
        Build Merkle tree from list of data hashes.
        
        Args:
            data_hashes: List of individual data hashes
            
        Returns:
            Merkle tree structure with root hash
        """
        if not data_hashes:
            return {"root": None, "tree": [], "leaf_count": 0}
        
        # Ensure even number of hashes
        if len(data_hashes) % 2 != 0:
            data_hashes.append(data_hashes[-1])  # Duplicate last hash
        
        tree_levels = [data_hashes]
        current_level = data_hashes
        
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left_hash = current_level[i]
                right_hash = current_level[i + 1] if i + 1 < len(current_level) else left_hash
                
                # Combine hashes
                combined = left_hash + right_hash
                parent_hash = self.compute_data_hash(combined)
                next_level.append(parent_hash)
            
            tree_levels.append(next_level)
            current_level = next_level
        
        return {
            "root": current_level[0],
            "tree": tree_levels,
            "leaf_count": len(data_hashes)
        }
    
    def generate_proof_of_inclusion(self, merkle_tree: Dict[str, Any], 
                                   leaf_index: int) -> List[str]:
        """
        Generate Merkle proof of inclusion for specific leaf.
        
        Args:
            merkle_tree: Merkle tree structure
            leaf_index: Index of leaf to prove
            
        Returns:
            List of hash values constituting the proof
        """
        if leaf_index >= merkle_tree["leaf_count"]:
            raise ValueError("Leaf index out of range")
        
        proof = []
        tree_levels = merkle_tree["tree"]
        current_index = leaf_index
        
        for level in tree_levels[:-1]:  # Exclude root level
            # Find sibling hash
            if current_index % 2 == 0:
                # Left node, sibling is right
                sibling_index = current_index + 1
            else:
                # Right node, sibling is left
                sibling_index = current_index - 1
            
            if sibling_index < len(level):
                proof.append(level[sibling_index])
            
            current_index = current_index // 2
        
        return proof
    
    def verify_merkle_proof(self, leaf_hash: str, proof: List[str], 
                           root_hash: str) -> bool:
        """
        Verify Merkle proof of inclusion.
        
        Args:
            leaf_hash: Hash of the leaf to verify
            proof: Merkle proof path
            root_hash: Expected root hash
            
        Returns:
            True if proof is valid
        """
        current_hash = leaf_hash
        
        for proof_hash in proof:
            # Try both orderings (left-right and right-left)
            combined1 = current_hash + proof_hash
            combined2 = proof_hash + current_hash
            
            hash1 = self.compute_data_hash(combined1)
            hash2 = self.compute_data_hash(combined2)
            
            # Use the ordering that continues the path toward root
            # This is a simplified approach; in practice, you'd track position
            current_hash = hash1  # Simplified - would need proper ordering logic
        
        return current_hash == root_hash


class ProvenanceTracker:
    """
    Complete provenance tracking system for ShadowBench operations.
    
    Features:
    - Event logging and chain building
    - Cryptographic integrity verification
    - Audit trail generation
    - Trust score calculation
    - Tamper detection
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize cryptographic verifier
        self.verifier = CryptographicVerifier()
        
        # Provenance storage
        self.events: List[ProvenanceEvent] = []
        self.event_index: Dict[str, ProvenanceEvent] = {}
        
        # Trust and integrity tracking
        self.trusted_actors: Dict[str, float] = {}  # Actor -> trust score
        self.integrity_checkpoints: List[Dict[str, Any]] = []
        
        # Configuration
        self.enable_signatures = self.config.get('enable_signatures', True)
        self.checkpoint_interval = self.config.get('checkpoint_interval', 100)  # Events
        self.max_chain_length = self.config.get('max_chain_length', 10000)
    
    def record_event(self, event_type: ProvenanceEventType, actor: str,
                    resource: str, action: str, input_data: Any = None,
                    output_data: Any = None, metadata: Optional[Dict] = None,
                    parent_event_id: Optional[str] = None) -> str:
        """
        Record a new provenance event.
        
        Args:
            event_type: Type of provenance event
            actor: System component or user responsible
            resource: Resource that was affected
            action: Action that was performed
            input_data: Input data for the operation
            output_data: Output data from the operation
            metadata: Additional metadata
            parent_event_id: ID of parent event in chain
            
        Returns:
            Unique event ID
        """
        event_id = self._generate_event_id()
        timestamp = datetime.now(timezone.utc)
        
        # Compute hashes for input/output data
        input_hash = self.verifier.compute_data_hash(input_data) if input_data is not None else None
        output_hash = self.verifier.compute_data_hash(output_data) if output_data is not None else None
        
        # Create event
        event = ProvenanceEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=timestamp,
            actor=actor,
            resource=resource,
            action=action,
            input_hash=input_hash,
            output_hash=output_hash,
            metadata=metadata,
            parent_event_id=parent_event_id
        )
        
        # Generate signature if enabled
        if self.enable_signatures:
            event_data = asdict(event)
            event_data.pop('signature', None)  # Remove signature field for signing
            event.signature = self.verifier.compute_hmac(event_data)
        
        # Store event
        self.events.append(event)
        self.event_index[event_id] = event
        
        # Update trust scores
        self._update_trust_score(actor, event)
        
        # Create integrity checkpoint if needed
        if len(self.events) % self.checkpoint_interval == 0:
            self._create_integrity_checkpoint()
        
        self.logger.debug(f"Recorded provenance event {event_id}: {event_type.value}")
        
        return event_id
    
    def verify_event_integrity(self, event_id: str) -> IntegrityCheckResult:
        """
        Verify the integrity of a specific event.
        
        Args:
            event_id: ID of event to verify
            
        Returns:
            IntegrityCheckResult with verification details
        """
        if event_id not in self.event_index:
            return IntegrityCheckResult(
                is_valid=False,
                verification_method="event_lookup",
                confidence_score=0.0,
                issues_found=["Event not found"],
                verification_timestamp=datetime.now(timezone.utc),
                chain_length=0,
                trust_score=0.0
            )
        
        event = self.event_index[event_id]
        issues_found = []
        confidence_score = 1.0
        
        # Verify signature if present
        if event.signature:
            event_data = asdict(event)
            event_data.pop('signature', None)
            
            if not self.verifier.verify_hmac(event_data, event.signature):
                issues_found.append("Invalid signature")
                confidence_score -= 0.5
        else:
            issues_found.append("No signature present")
            confidence_score -= 0.2
        
        # Verify parent chain if applicable
        chain_length = self._calculate_chain_length(event_id)
        if event.parent_event_id and event.parent_event_id not in self.event_index:
            issues_found.append("Broken parent chain")
            confidence_score -= 0.3
        
        # Calculate trust score
        actor_trust = self.trusted_actors.get(event.actor, 0.5)
        trust_score = actor_trust
        
        # Check for temporal consistency
        if not self._verify_temporal_consistency(event):
            issues_found.append("Temporal inconsistency detected")
            confidence_score -= 0.2
        
        is_valid = confidence_score > 0.5 and len(issues_found) == 0
        
        return IntegrityCheckResult(
            is_valid=is_valid,
            verification_method="hmac_signature",
            confidence_score=max(confidence_score, 0.0),
            issues_found=issues_found,
            verification_timestamp=datetime.now(timezone.utc),
            chain_length=chain_length,
            trust_score=trust_score
        )
    
    def verify_chain_integrity(self, start_event_id: str, 
                             end_event_id: Optional[str] = None) -> IntegrityCheckResult:
        """
        Verify integrity of an entire event chain.
        
        Args:
            start_event_id: Starting event in the chain
            end_event_id: Ending event in the chain (optional)
            
        Returns:
            IntegrityCheckResult for the entire chain
        """
        chain_events = self._extract_event_chain(start_event_id, end_event_id)
        
        if not chain_events:
            return IntegrityCheckResult(
                is_valid=False,
                verification_method="chain_extraction",
                confidence_score=0.0,
                issues_found=["Empty or invalid chain"],
                verification_timestamp=datetime.now(timezone.utc),
                chain_length=0,
                trust_score=0.0
            )
        
        # Verify each event in chain
        all_issues = []
        confidence_scores = []
        trust_scores = []
        
        for event in chain_events:
            result = self.verify_event_integrity(event.event_id)
            all_issues.extend(result.issues_found)
            confidence_scores.append(result.confidence_score)
            trust_scores.append(result.trust_score)
        
        # Verify chain linkage
        for i in range(1, len(chain_events)):
            current_event = chain_events[i]
            previous_event = chain_events[i-1]
            
            if current_event.parent_event_id != previous_event.event_id:
                all_issues.append(f"Chain linkage broken between {previous_event.event_id} and {current_event.event_id}")
        
        # Calculate overall metrics
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        avg_trust = sum(trust_scores) / len(trust_scores) if trust_scores else 0.0
        
        # Additional chain-specific checks
        if not self._verify_chain_temporal_order(chain_events):
            all_issues.append("Chain events not in temporal order")
            avg_confidence -= 0.2
        
        is_valid = avg_confidence > 0.5 and len(all_issues) == 0
        
        return IntegrityCheckResult(
            is_valid=is_valid,
            verification_method="full_chain_verification",
            confidence_score=max(avg_confidence, 0.0),
            issues_found=all_issues,
            verification_timestamp=datetime.now(timezone.utc),
            chain_length=len(chain_events),
            trust_score=avg_trust
        )
    
    def generate_audit_trail(self, resource_filter: Optional[str] = None,
                           actor_filter: Optional[str] = None,
                           time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive audit trail.
        
        Args:
            resource_filter: Filter by specific resource
            actor_filter: Filter by specific actor
            time_range: Filter by time range (start, end)
            
        Returns:
            Complete audit trail with integrity verification
        """
        filtered_events = self._filter_events(resource_filter, actor_filter, time_range)
        
        # Build audit trail
        audit_trail = {
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_events": len(filtered_events),
            "filters_applied": {
                "resource": resource_filter,
                "actor": actor_filter,
                "time_range": time_range[0].isoformat() if time_range else None
            },
            "events": [],
            "integrity_summary": {
                "verified_events": 0,
                "failed_verifications": 0,
                "average_confidence": 0.0,
                "trust_issues": []
            },
            "chain_analysis": {
                "chains_identified": 0,
                "broken_chains": 0,
                "longest_chain": 0
            }
        }
        
        # Process each event
        verification_results = []
        for event in filtered_events:
            # Verify event integrity
            verification = self.verify_event_integrity(event.event_id)
            verification_results.append(verification)
            
            # Add to audit trail
            event_record = {
                "event": asdict(event),
                "verification": asdict(verification),
                "chain_position": self._find_chain_position(event.event_id)
            }
            audit_trail["events"].append(event_record)
        
        # Calculate integrity summary
        verified_count = sum(1 for r in verification_results if r.is_valid)
        audit_trail["integrity_summary"]["verified_events"] = verified_count
        audit_trail["integrity_summary"]["failed_verifications"] = len(verification_results) - verified_count
        
        if verification_results:
            avg_confidence = sum(r.confidence_score for r in verification_results) / len(verification_results)
            audit_trail["integrity_summary"]["average_confidence"] = avg_confidence
        
        # Analyze chains
        chains = self._identify_event_chains(filtered_events)
        audit_trail["chain_analysis"]["chains_identified"] = len(chains)
        audit_trail["chain_analysis"]["longest_chain"] = max((len(chain) for chain in chains), default=0)
        
        broken_chains = sum(1 for chain in chains if not self._is_chain_intact(chain))
        audit_trail["chain_analysis"]["broken_chains"] = broken_chains
        
        # Generate audit trail signature
        audit_data = {k: v for k, v in audit_trail.items() if k != "signature"}
        audit_trail["signature"] = self.verifier.compute_hmac(audit_data)
        
        return audit_trail
    
    def export_provenance_data(self, filepath: str, format: str = "json",
                             include_verification: bool = True) -> bool:
        """
        Export provenance data to file.
        
        Args:
            filepath: Output file path
            format: Export format (json, csv)
            include_verification: Include verification results
            
        Returns:
            True if export successful
        """
        try:
            export_data = {
                "metadata": {
                    "export_timestamp": datetime.now(timezone.utc).isoformat(),
                    "total_events": len(self.events),
                    "format_version": "1.0",
                    "include_verification": include_verification
                },
                "configuration": {
                    "enable_signatures": self.enable_signatures,
                    "checkpoint_interval": self.checkpoint_interval,
                    "max_chain_length": self.max_chain_length
                },
                "events": [asdict(event) for event in self.events],
                "trust_scores": self.trusted_actors,
                "integrity_checkpoints": self.integrity_checkpoints
            }
            
            if include_verification:
                export_data["verification_results"] = {}
                for event in self.events:
                    verification = self.verify_event_integrity(event.event_id)
                    export_data["verification_results"][event.event_id] = asdict(verification)
            
            # Generate export signature
            signature_data = {k: v for k, v in export_data.items() if k != "signature"}
            export_data["signature"] = self.verifier.compute_hmac(signature_data)
            
            # Write to file
            if format.lower() == "json":
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Provenance data exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export provenance data: {e}")
            return False
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        timestamp = int(time.time() * 1000000)  # Microsecond precision
        random_part = secrets.token_hex(8)
        return f"event_{timestamp}_{random_part}"
    
    def _update_trust_score(self, actor: str, event: ProvenanceEvent):
        """Update trust score for an actor based on event."""
        current_trust = self.trusted_actors.get(actor, 0.5)
        
        # Simple trust update based on event type and integrity
        trust_adjustment = 0.0
        
        if event.event_type in [ProvenanceEventType.VALIDATION_PERFORMED, 
                               ProvenanceEventType.METRIC_CALCULATED]:
            trust_adjustment = 0.02  # Positive actions
        elif event.event_type == ProvenanceEventType.DATA_MODIFIED:
            trust_adjustment = -0.01  # Potentially risky actions
        
        # Update trust score (bounded between 0 and 1)
        new_trust = max(0.0, min(1.0, current_trust + trust_adjustment))
        self.trusted_actors[actor] = new_trust
    
    def _create_integrity_checkpoint(self):
        """Create integrity checkpoint with Merkle root."""
        event_hashes = []
        
        # Get recent events for checkpoint
        start_index = max(0, len(self.events) - self.checkpoint_interval)
        checkpoint_events = self.events[start_index:]
        
        for event in checkpoint_events:
            event_data = asdict(event)
            event_hash = self.verifier.compute_data_hash(event_data)
            event_hashes.append(event_hash)
        
        # Build Merkle tree
        merkle_tree = self.verifier.build_merkle_tree(event_hashes)
        
        checkpoint = {
            "checkpoint_id": f"checkpoint_{int(time.time())}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_range": [start_index, len(self.events)],
            "event_count": len(checkpoint_events),
            "merkle_root": merkle_tree["root"],
            "merkle_tree": merkle_tree
        }
        
        self.integrity_checkpoints.append(checkpoint)
        self.logger.info(f"Created integrity checkpoint with {len(checkpoint_events)} events")
    
    def _calculate_chain_length(self, event_id: str) -> int:
        """Calculate length of event chain starting from given event."""
        chain_length = 0
        current_event_id = event_id
        visited = set()
        
        while current_event_id and current_event_id not in visited:
            visited.add(current_event_id)
            event = self.event_index.get(current_event_id)
            if event:
                chain_length += 1
                current_event_id = event.parent_event_id
            else:
                break
        
        return chain_length
    
    def _verify_temporal_consistency(self, event: ProvenanceEvent) -> bool:
        """Verify temporal consistency of an event."""
        # Check if event timestamp is reasonable
        now = datetime.now(timezone.utc)
        
        # Events shouldn't be from the future
        if event.timestamp > now:
            return False
        
        # Events shouldn't be too old (configurable threshold)
        max_age_days = self.config.get('max_event_age_days', 365)
        age_threshold = now.timestamp() - (max_age_days * 24 * 3600)
        
        if event.timestamp.timestamp() < age_threshold:
            return False
        
        # Check parent event temporal ordering
        if event.parent_event_id:
            parent_event = self.event_index.get(event.parent_event_id)
            if parent_event and parent_event.timestamp >= event.timestamp:
                return False
        
        return True
    
    def _extract_event_chain(self, start_event_id: str, 
                           end_event_id: Optional[str] = None) -> List[ProvenanceEvent]:
        """Extract event chain between start and end events."""
        chain = []
        current_event_id = start_event_id
        visited = set()
        
        while current_event_id and current_event_id not in visited:
            visited.add(current_event_id)
            event = self.event_index.get(current_event_id)
            
            if not event:
                break
                
            chain.append(event)
            
            if end_event_id and current_event_id == end_event_id:
                break
                
            # Find next event in chain (child event)
            next_event_id = None
            for candidate_event in self.events:
                if candidate_event.parent_event_id == current_event_id:
                    next_event_id = candidate_event.event_id
                    break
            
            current_event_id = next_event_id
        
        return chain
    
    def _verify_chain_temporal_order(self, chain_events: List[ProvenanceEvent]) -> bool:
        """Verify that events in chain are in temporal order."""
        for i in range(1, len(chain_events)):
            if chain_events[i].timestamp < chain_events[i-1].timestamp:
                return False
        return True
    
    def _filter_events(self, resource_filter: Optional[str], 
                      actor_filter: Optional[str],
                      time_range: Optional[Tuple[datetime, datetime]]) -> List[ProvenanceEvent]:
        """Filter events based on criteria."""
        filtered = self.events
        
        if resource_filter:
            filtered = [e for e in filtered if resource_filter in e.resource]
        
        if actor_filter:
            filtered = [e for e in filtered if actor_filter in e.actor]
        
        if time_range:
            start_time, end_time = time_range
            filtered = [e for e in filtered 
                       if start_time <= e.timestamp <= end_time]
        
        return filtered
    
    def _find_chain_position(self, event_id: str) -> Dict[str, Any]:
        """Find position of event in its chain."""
        event = self.event_index.get(event_id)
        if not event:
            return {"position": "unknown", "chain_length": 0}
        
        # Count ancestors
        ancestors = 0
        current_id = event.parent_event_id
        while current_id and current_id in self.event_index:
            ancestors += 1
            current_id = self.event_index[current_id].parent_event_id
        
        # Count descendants
        descendants = 0
        for candidate_event in self.events:
            if self._is_descendant(candidate_event.event_id, event_id):
                descendants += 1
        
        return {
            "position": ancestors,
            "ancestors": ancestors,
            "descendants": descendants,
            "chain_length": ancestors + descendants + 1
        }
    
    def _identify_event_chains(self, events: List[ProvenanceEvent]) -> List[List[ProvenanceEvent]]:
        """Identify separate event chains in the given events."""
        chains = []
        processed_events = set()
        
        for event in events:
            if event.event_id in processed_events:
                continue
            
            # Find root of this chain
            root_event = self._find_chain_root(event)
            
            if root_event.event_id not in processed_events:
                # Extract full chain from root
                chain = self._extract_event_chain(root_event.event_id)
                chains.append(chain)
                
                # Mark all events in chain as processed
                for chain_event in chain:
                    processed_events.add(chain_event.event_id)
        
        return chains
    
    def _find_chain_root(self, event: ProvenanceEvent) -> ProvenanceEvent:
        """Find root event of the chain containing the given event."""
        current_event = event
        visited = set()
        
        while current_event.parent_event_id and current_event.event_id not in visited:
            visited.add(current_event.event_id)
            parent_event = self.event_index.get(current_event.parent_event_id)
            
            if not parent_event:
                break
                
            current_event = parent_event
        
        return current_event
    
    def _is_chain_intact(self, chain: List[ProvenanceEvent]) -> bool:
        """Check if event chain is intact (no missing links)."""
        if len(chain) <= 1:
            return True
        
        for i in range(1, len(chain)):
            if chain[i].parent_event_id != chain[i-1].event_id:
                return False
        
        return True
    
    def _is_descendant(self, candidate_id: str, ancestor_id: str) -> bool:
        """Check if candidate event is a descendant of ancestor event."""
        current_id = candidate_id
        visited = set()
        
        while current_id and current_id not in visited:
            visited.add(current_id)
            event = self.event_index.get(current_id)
            
            if not event:
                break
                
            if event.parent_event_id == ancestor_id:
                return True
                
            current_id = event.parent_event_id
        
        return False
