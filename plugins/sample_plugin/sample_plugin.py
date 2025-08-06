"""
Sample Plugin for ShadowBench
Demonstrates the plugin architecture with a simple attack generator.
"""

from plugins import AttackGeneratorPlugin, PluginMetadata, PluginType
import numpy as np
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone


class SampleAttackPlugin(AttackGeneratorPlugin):
    """
    Sample attack generator plugin for demonstration.
    
    This plugin implements a simple text-based adversarial attack
    that introduces character substitutions and word replacements.
    """
    
    def __init__(self, plugin_id: str, config: Optional[Dict] = None):
        super().__init__(plugin_id, config)
        
        # Plugin-specific configuration
        self.attack_intensity = self.config.get('attack_intensity', 0.1)  # 10% of characters
        self.character_substitutions = self.config.get('character_substitutions', {
            'a': '@', 'e': '3', 'i': '1', 'o': '0', 's': '$', 't': '7'
        })
        self.word_replacements = self.config.get('word_replacements', {
            'good': 'bad', 'safe': 'unsafe', 'secure': 'vulnerable',
            'trust': 'doubt', 'verify': 'ignore', 'clean': 'malicious'
        })
        
        self.logger.info(f"Sample Attack Plugin initialized with intensity {self.attack_intensity}")
    
    def initialize(self) -> bool:
        """Initialize the plugin."""
        try:
            # Validate configuration
            if not 0.0 <= self.attack_intensity <= 1.0:
                self.logger.error("Attack intensity must be between 0.0 and 1.0")
                return False
            
            if not isinstance(self.character_substitutions, dict):
                self.logger.error("Character substitutions must be a dictionary")
                return False
            
            self.is_initialized = True
            self.logger.info("Sample Attack Plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize plugin: {e}")
            return False
    
    def execute(self, input_data: Any, context: Optional[Dict] = None) -> Any:
        """Execute the plugin's main functionality."""
        context = context or {}
        attack_config = context.get('attack_config', {})
        
        return self.generate_attack(input_data, attack_config)
    
    def generate_attack(self, target_data: Any, attack_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate adversarial attack on text data.
        
        Args:
            target_data: Text to attack
            attack_config: Attack configuration
            
        Returns:
            Attack result with perturbed data and metadata
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Validate input
            if not isinstance(target_data, str):
                if hasattr(target_data, '__str__'):
                    target_text = str(target_data)
                else:
                    raise ValueError("Target data must be text or convertible to string")
            else:
                target_text = target_data
            
            # Apply attack configuration
            local_intensity = attack_config.get('intensity', self.attack_intensity)
            local_substitutions = attack_config.get('substitutions', self.character_substitutions)
            local_word_replacements = attack_config.get('word_replacements', self.word_replacements)
            
            # Generate attacked text
            attacked_text = self._apply_character_attack(target_text, local_intensity, local_substitutions)
            attacked_text = self._apply_word_attack(attacked_text, local_word_replacements)
            
            # Calculate metrics
            char_changes = sum(1 for i, (orig, attack) in enumerate(zip(target_text, attacked_text))
                              if orig != attack and i < len(attacked_text))
            change_ratio = char_changes / len(target_text) if target_text else 0.0
            
            # Calculate semantic similarity (simplified)
            semantic_similarity = self._calculate_semantic_similarity(target_text, attacked_text)
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            attack_result = {
                "original_text": target_text,
                "attacked_text": attacked_text,
                "attack_success": change_ratio > 0.0,
                "metrics": {
                    "character_changes": char_changes,
                    "change_ratio": change_ratio,
                    "semantic_similarity": semantic_similarity,
                    "text_length_original": len(target_text),
                    "text_length_attacked": len(attacked_text),
                    "execution_time": execution_time
                },
                "attack_config": {
                    "intensity": local_intensity,
                    "substitutions_applied": len(local_substitutions),
                    "word_replacements_applied": len(local_word_replacements)
                },
                "attack_metadata": {
                    "plugin_id": self.plugin_id,
                    "attack_type": "character_and_word_substitution",
                    "timestamp": start_time.isoformat(),
                    "version": "1.0.0"
                }
            }
            
            return attack_result
            
        except Exception as e:
            self.logger.error(f"Attack generation failed: {e}")
            return {
                "original_text": target_data if isinstance(target_data, str) else str(target_data),
                "attacked_text": None,
                "attack_success": False,
                "error": str(e),
                "metrics": {},
                "attack_metadata": {
                    "plugin_id": self.plugin_id,
                    "timestamp": start_time.isoformat(),
                    "failed": True
                }
            }
    
    def cleanup(self) -> bool:
        """Clean up plugin resources."""
        try:
            # No specific cleanup needed for this simple plugin
            self.logger.info("Sample Attack Plugin cleaned up successfully")
            return True
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return False
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            plugin_id=self.plugin_id,
            name="Sample Attack Plugin",
            version="1.0.0",
            description="Sample text-based adversarial attack plugin for demonstration",
            author="ShadowBench Team",
            plugin_type=PluginType.ATTACK_GENERATOR,
            dependencies=[],
            api_version="1.0",
            entry_point="sample_plugin.py",
            configuration_schema={
                "type": "object",
                "properties": {
                    "attack_intensity": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Ratio of characters to modify"
                    },
                    "character_substitutions": {
                        "type": "object",
                        "description": "Character substitution mappings"
                    },
                    "word_replacements": {
                        "type": "object",
                        "description": "Word replacement mappings"
                    }
                }
            },
            permissions=["data_modify"]
        )
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data format."""
        try:
            # Accept strings or objects that can be converted to strings
            if isinstance(input_data, str):
                return True
            elif hasattr(input_data, '__str__'):
                return True
            else:
                self.logger.warning(f"Invalid input data type: {type(input_data)}")
                return False
        except Exception as e:
            self.logger.error(f"Input validation error: {e}")
            return False
    
    def _apply_character_attack(self, text: str, intensity: float, 
                               substitutions: Dict[str, str]) -> str:
        """Apply character-level substitution attack."""
        if not substitutions or intensity <= 0:
            return text
        
        attacked_text = list(text.lower())
        total_chars = len(attacked_text)
        chars_to_modify = int(total_chars * intensity)
        
        # Randomly select characters to modify
        if chars_to_modify > 0:
            import random
            random.seed(42)  # For reproducible results
            
            positions = random.sample(range(total_chars), 
                                    min(chars_to_modify, total_chars))
            
            for pos in positions:
                char = attacked_text[pos]
                if char in substitutions:
                    attacked_text[pos] = substitutions[char]
        
        return ''.join(attacked_text)
    
    def _apply_word_attack(self, text: str, word_replacements: Dict[str, str]) -> str:
        """Apply word-level replacement attack."""
        if not word_replacements:
            return text
        
        words = text.split()
        attacked_words = []
        
        for word in words:
            # Clean word for matching (remove punctuation)
            clean_word = word.strip('.,!?;:"()[]{}').lower()
            
            if clean_word in word_replacements:
                # Replace while preserving punctuation
                replacement = word_replacements[clean_word]
                if word != clean_word:
                    # Preserve original casing and punctuation
                    prefix = word[:len(word) - len(word.lstrip('.,!?;:"()[]{}'))]
                    suffix = word[len(clean_word) + len(prefix):]
                    attacked_word = prefix + replacement + suffix
                else:
                    attacked_word = replacement
                
                attacked_words.append(attacked_word)
            else:
                attacked_words.append(word)
        
        return ' '.join(attacked_words)
    
    def _calculate_semantic_similarity(self, original: str, attacked: str) -> float:
        """
        Calculate semantic similarity between original and attacked text.
        Simplified implementation using character-level similarity.
        """
        if not original or not attacked:
            return 0.0
        
        # Simple character-based similarity
        original_chars = set(original.lower())
        attacked_chars = set(attacked.lower())
        
        if not original_chars:
            return 1.0 if not attacked_chars else 0.0
        
        intersection = len(original_chars.intersection(attacked_chars))
        union = len(original_chars.union(attacked_chars))
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Also consider length similarity
        length_similarity = min(len(original), len(attacked)) / max(len(original), len(attacked))
        
        # Weighted combination
        semantic_similarity = 0.7 * jaccard_similarity + 0.3 * length_similarity
        
        return semantic_similarity


# Plugin factory function for dynamic loading
def create_plugin(plugin_id: str, config: Optional[Dict] = None) -> SampleAttackPlugin:
    """Factory function to create the plugin instance."""
    return SampleAttackPlugin(plugin_id, config)


# Example usage and testing
if __name__ == "__main__":
    # Example of how the plugin would be used
    logging.basicConfig(level=logging.INFO)
    
    # Create plugin instance
    plugin = SampleAttackPlugin("sample_attack", {
        "attack_intensity": 0.2,
        "character_substitutions": {'a': '@', 'e': '3', 'i': '1'},
        "word_replacements": {'good': 'bad', 'safe': 'dangerous'}
    })
    
    # Initialize plugin
    if plugin.initialize():
        # Test attack generation
        test_text = "This is a good and safe example text for testing."
        attack_config = {"intensity": 0.15}
        
        result = plugin.generate_attack(test_text, attack_config)
        
        print("Attack Result:")
        print(f"Original: {result['original_text']}")
        print(f"Attacked: {result['attacked_text']}")
        print(f"Success: {result['attack_success']}")
        print(f"Metrics: {json.dumps(result['metrics'], indent=2)}")
        
        # Cleanup
        plugin.cleanup()
    else:
        print("Plugin initialization failed")
