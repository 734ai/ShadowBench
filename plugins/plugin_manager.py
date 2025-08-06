"""
Dynamic Plugin Architecture for ShadowBench
Provides extensible plugin system with runtime loading and REST API integration.
"""

import importlib
import importlib.util
import inspect
import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Callable, Type, Union
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
import aiohttp
import threading
import time
from datetime import datetime, timezone


class PluginType(Enum):
    """Types of plugins supported by ShadowBench."""
    ATTACK_GENERATOR = "attack_generator"
    METRIC_CALCULATOR = "metric_calculator" 
    MODEL_WRAPPER = "model_wrapper"
    DATA_PROCESSOR = "data_processor"
    RESULT_FORMATTER = "result_formatter"
    VALIDATION_HOOK = "validation_hook"
    MIDDLEWARE = "middleware"
    EXTERNAL_API = "external_api"


class PluginStatus(Enum):
    """Plugin status states."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class PluginMetadata:
    """Plugin metadata information."""
    plugin_id: str
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str]
    api_version: str
    entry_point: str
    configuration_schema: Dict[str, Any]
    permissions: List[str]
    status: PluginStatus = PluginStatus.UNLOADED
    load_timestamp: Optional[datetime] = None
    error_message: Optional[str] = None


class ShadowBenchPlugin(ABC):
    """
    Abstract base class for all ShadowBench plugins.
    
    All plugins must inherit from this class and implement the required methods.
    """
    
    def __init__(self, plugin_id: str, config: Optional[Dict] = None):
        self.plugin_id = plugin_id
        self.config = config or {}
        self.logger = logging.getLogger(f"plugin.{plugin_id}")
        self.is_initialized = False
        self.performance_metrics = {
            "total_calls": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0,
            "error_count": 0,
            "last_call_timestamp": None
        }
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the plugin.
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    def execute(self, input_data: Any, context: Optional[Dict] = None) -> Any:
        """
        Execute the plugin's main functionality.
        
        Args:
            input_data: Input data for the plugin
            context: Execution context
            
        Returns:
            Plugin output
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> bool:
        """
        Clean up plugin resources.
        
        Returns:
            True if cleanup successful
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """
        Get plugin metadata.
        
        Returns:
            Plugin metadata
        """
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data format.
        
        Args:
            input_data: Data to validate
            
        Returns:
            True if valid
        """
        return True  # Default implementation
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get plugin performance metrics."""
        return self.performance_metrics.copy()
    
    def _update_metrics(self, execution_time: float, success: bool):
        """Update performance metrics."""
        self.performance_metrics["total_calls"] += 1
        self.performance_metrics["total_execution_time"] += execution_time
        self.performance_metrics["average_execution_time"] = (
            self.performance_metrics["total_execution_time"] / 
            self.performance_metrics["total_calls"]
        )
        self.performance_metrics["last_call_timestamp"] = datetime.now(timezone.utc)
        
        if not success:
            self.performance_metrics["error_count"] += 1


class AttackGeneratorPlugin(ShadowBenchPlugin):
    """Base class for attack generator plugins."""
    
    @abstractmethod
    def generate_attack(self, target_data: Any, attack_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate adversarial attack.
        
        Args:
            target_data: Data to attack
            attack_config: Attack configuration
            
        Returns:
            Attack result with perturbed data and metadata
        """
        pass


class MetricCalculatorPlugin(ShadowBenchPlugin):
    """Base class for metric calculator plugins."""
    
    @abstractmethod
    def calculate_metric(self, ground_truth: Any, predictions: Any, 
                        context: Optional[Dict] = None) -> Dict[str, float]:
        """
        Calculate metrics.
        
        Args:
            ground_truth: Ground truth data
            predictions: Model predictions
            context: Additional context
            
        Returns:
            Dictionary of metric name to value
        """
        pass


class ModelWrapperPlugin(ShadowBenchPlugin):
    """Base class for model wrapper plugins."""
    
    @abstractmethod
    def query_model(self, input_data: Any, **kwargs) -> Any:
        """
        Query the wrapped model.
        
        Args:
            input_data: Input for the model
            **kwargs: Additional parameters
            
        Returns:
            Model response
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Model metadata and capabilities
        """
        pass


class PluginRegistry:
    """
    Registry for managing plugin discovery and metadata.
    
    Features:
    - Plugin discovery from directories and remote repositories
    - Metadata validation and dependency resolution  
    - Plugin versioning and conflict detection
    - Security scanning and permission management
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Plugin storage
        self.plugins: Dict[str, PluginMetadata] = {}
        self.plugin_dependencies: Dict[str, List[str]] = {}
        
        # Discovery paths
        self.plugin_directories: List[Path] = []
        self.remote_repositories: List[str] = []
        
        # Security and validation
        self.allowed_permissions = {
            "file_read", "file_write", "network_access", 
            "model_query", "data_modify", "system_info"
        }
        self.security_scanner_enabled = self.config.get('security_scanning', True)
        
        self._initialize_registry()
    
    def _initialize_registry(self):
        """Initialize plugin registry."""
        # Add default plugin directories
        default_dirs = [
            Path("./plugins"),
            Path("./plugins/community"),
            Path("./plugins/official")
        ]
        
        for dir_path in default_dirs:
            if dir_path.exists():
                self.plugin_directories.append(dir_path)
        
        # Add configured plugin directories
        config_dirs = self.config.get('plugin_directories', [])
        for dir_str in config_dirs:
            dir_path = Path(dir_str)
            if dir_path.exists():
                self.plugin_directories.append(dir_path)
        
        # Add remote repositories
        self.remote_repositories.extend(
            self.config.get('remote_repositories', [])
        )
        
        self.logger.info(f"Plugin registry initialized with {len(self.plugin_directories)} local directories")
    
    def discover_plugins(self, force_refresh: bool = False) -> List[PluginMetadata]:
        """
        Discover all available plugins.
        
        Args:
            force_refresh: Force re-discovery even if cached
            
        Returns:
            List of discovered plugin metadata
        """
        if not force_refresh and self.plugins:
            return list(self.plugins.values())
        
        discovered_plugins = []
        
        # Discover from local directories
        for plugin_dir in self.plugin_directories:
            discovered_plugins.extend(self._discover_local_plugins(plugin_dir))
        
        # Discover from remote repositories (if enabled)
        if self.config.get('enable_remote_discovery', False):
            for repo_url in self.remote_repositories:
                try:
                    remote_plugins = asyncio.run(self._discover_remote_plugins(repo_url))
                    discovered_plugins.extend(remote_plugins)
                except Exception as e:
                    self.logger.error(f"Failed to discover plugins from {repo_url}: {e}")
        
        # Update registry
        for plugin_metadata in discovered_plugins:
            self.plugins[plugin_metadata.plugin_id] = plugin_metadata
        
        # Resolve dependencies
        self._resolve_dependencies()
        
        self.logger.info(f"Discovered {len(discovered_plugins)} plugins")
        return discovered_plugins
    
    def _discover_local_plugins(self, directory: Path) -> List[PluginMetadata]:
        """Discover plugins in local directory."""
        plugins = []
        
        for item in directory.iterdir():
            if item.is_dir():
                # Check for plugin manifest
                manifest_file = item / "plugin.yaml"
                if not manifest_file.exists():
                    manifest_file = item / "plugin.json"
                
                if manifest_file.exists():
                    try:
                        plugin_metadata = self._load_plugin_manifest(manifest_file)
                        
                        # Security scan if enabled
                        if self.security_scanner_enabled:
                            if self._security_scan_plugin(item):
                                plugins.append(plugin_metadata)
                            else:
                                self.logger.warning(f"Plugin {plugin_metadata.plugin_id} failed security scan")
                        else:
                            plugins.append(plugin_metadata)
                            
                    except Exception as e:
                        self.logger.error(f"Failed to load plugin from {item}: {e}")
        
        return plugins
    
    async def _discover_remote_plugins(self, repository_url: str) -> List[PluginMetadata]:
        """Discover plugins from remote repository."""
        plugins = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch repository index
                index_url = f"{repository_url}/index.json"
                async with session.get(index_url) as response:
                    if response.status == 200:
                        index_data = await response.json()
                        
                        for plugin_info in index_data.get("plugins", []):
                            try:
                                # Fetch plugin manifest
                                manifest_url = f"{repository_url}/{plugin_info['path']}/plugin.json"
                                async with session.get(manifest_url) as manifest_response:
                                    if manifest_response.status == 200:
                                        manifest_data = await manifest_response.json()
                                        plugin_metadata = self._parse_plugin_manifest(manifest_data, repository_url)
                                        plugins.append(plugin_metadata)
                                        
                            except Exception as e:
                                self.logger.error(f"Failed to fetch remote plugin {plugin_info}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to access remote repository {repository_url}: {e}")
        
        return plugins
    
    def _load_plugin_manifest(self, manifest_path: Path) -> PluginMetadata:
        """Load plugin manifest from file."""
        if manifest_path.suffix == '.yaml':
            with open(manifest_path, 'r') as f:
                manifest_data = yaml.safe_load(f)
        else:  # JSON
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
        
        return self._parse_plugin_manifest(manifest_data, str(manifest_path.parent))
    
    def _parse_plugin_manifest(self, manifest_data: Dict[str, Any], 
                              plugin_path: str) -> PluginMetadata:
        """Parse plugin manifest data."""
        return PluginMetadata(
            plugin_id=manifest_data["plugin_id"],
            name=manifest_data["name"],
            version=manifest_data["version"],
            description=manifest_data.get("description", ""),
            author=manifest_data.get("author", "Unknown"),
            plugin_type=PluginType(manifest_data["plugin_type"]),
            dependencies=manifest_data.get("dependencies", []),
            api_version=manifest_data.get("api_version", "1.0"),
            entry_point=manifest_data["entry_point"],
            configuration_schema=manifest_data.get("configuration_schema", {}),
            permissions=manifest_data.get("permissions", [])
        )
    
    def _security_scan_plugin(self, plugin_path: Path) -> bool:
        """Perform security scan on plugin."""
        # Basic security checks
        
        # Check for suspicious file extensions
        suspicious_extensions = {'.exe', '.dll', '.so', '.bat', '.cmd', '.sh'}
        for file_path in plugin_path.rglob('*'):
            if file_path.suffix.lower() in suspicious_extensions:
                self.logger.warning(f"Suspicious file found: {file_path}")
                return False
        
        # Check for dangerous imports (simplified check)
        dangerous_imports = {'subprocess', 'os.system', 'eval', 'exec'}
        for python_file in plugin_path.glob('**/*.py'):
            try:
                with open(python_file, 'r') as f:
                    content = f.read()
                    for dangerous_import in dangerous_imports:
                        if dangerous_import in content:
                            self.logger.warning(f"Potentially dangerous import '{dangerous_import}' found in {python_file}")
                            # Don't fail completely, but log warning
            except Exception:
                pass  # Skip files that can't be read
        
        return True
    
    def _resolve_dependencies(self):
        """Resolve plugin dependencies."""
        for plugin_id, metadata in self.plugins.items():
            self.plugin_dependencies[plugin_id] = []
            
            for dependency in metadata.dependencies:
                if dependency in self.plugins:
                    self.plugin_dependencies[plugin_id].append(dependency)
                else:
                    self.logger.warning(f"Plugin {plugin_id} has unresolved dependency: {dependency}")
    
    def get_plugin(self, plugin_id: str) -> Optional[PluginMetadata]:
        """Get plugin metadata by ID."""
        return self.plugins.get(plugin_id)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginMetadata]:
        """Get all plugins of specific type."""
        return [plugin for plugin in self.plugins.values() 
                if plugin.plugin_type == plugin_type]
    
    def validate_plugin(self, plugin_id: str) -> Dict[str, Any]:
        """
        Validate plugin metadata and dependencies.
        
        Args:
            plugin_id: Plugin to validate
            
        Returns:
            Validation results
        """
        if plugin_id not in self.plugins:
            return {
                "is_valid": False,
                "errors": ["Plugin not found"],
                "warnings": []
            }
        
        plugin = self.plugins[plugin_id]
        errors = []
        warnings = []
        
        # Validate required fields
        required_fields = ['plugin_id', 'name', 'version', 'plugin_type', 'entry_point']
        for field in required_fields:
            if not getattr(plugin, field, None):
                errors.append(f"Missing required field: {field}")
        
        # Validate permissions
        for permission in plugin.permissions:
            if permission not in self.allowed_permissions:
                warnings.append(f"Unknown permission requested: {permission}")
        
        # Validate dependencies
        for dependency in plugin.dependencies:
            if dependency not in self.plugins:
                errors.append(f"Unresolved dependency: {dependency}")
        
        # Validate API version
        supported_versions = ["1.0", "1.1"]
        if plugin.api_version not in supported_versions:
            warnings.append(f"Unsupported API version: {plugin.api_version}")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }


class PluginManager:
    """
    Dynamic plugin manager with runtime loading and lifecycle management.
    
    Features:
    - Dynamic loading/unloading of plugins
    - Dependency injection and resolution
    - Plugin lifecycle management
    - Performance monitoring and health checks
    - Sandbox execution environment
    - Hot-reloading capabilities
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Plugin registry
        self.registry = PluginRegistry(self.config.get('registry', {}))
        
        # Loaded plugins
        self.loaded_plugins: Dict[str, ShadowBenchPlugin] = {}
        self.plugin_instances: Dict[str, Any] = {}  # Actual plugin instances
        
        # Plugin execution context
        self.execution_context = {
            "sandbox_enabled": self.config.get('sandbox_enabled', False),
            "max_execution_time": self.config.get('max_execution_time', 30.0),
            "memory_limit": self.config.get('memory_limit', 512 * 1024 * 1024)  # 512MB
        }
        
        # Performance monitoring
        self.performance_monitor = PluginPerformanceMonitor()
        
        # Hot-reload support
        self.hot_reload_enabled = self.config.get('hot_reload_enabled', False)
        self.file_watchers: Dict[str, Any] = {}
        
        self.logger.info("Plugin manager initialized")
    
    def discover_plugins(self) -> List[PluginMetadata]:
        """Discover all available plugins."""
        return self.registry.discover_plugins()
    
    def load_plugin(self, plugin_id: str, config: Optional[Dict] = None) -> bool:
        """
        Load a plugin by ID.
        
        Args:
            plugin_id: Plugin to load
            config: Plugin-specific configuration
            
        Returns:
            True if loading successful
        """
        if plugin_id in self.loaded_plugins:
            self.logger.warning(f"Plugin {plugin_id} already loaded")
            return True
        
        # Get plugin metadata
        metadata = self.registry.get_plugin(plugin_id)
        if not metadata:
            self.logger.error(f"Plugin {plugin_id} not found in registry")
            return False
        
        # Validate plugin
        validation = self.registry.validate_plugin(plugin_id)
        if not validation["is_valid"]:
            self.logger.error(f"Plugin {plugin_id} validation failed: {validation['errors']}")
            return False
        
        try:
            # Load dependencies first
            for dependency in metadata.dependencies:
                if dependency not in self.loaded_plugins:
                    if not self.load_plugin(dependency):
                        self.logger.error(f"Failed to load dependency {dependency} for {plugin_id}")
                        return False
            
            # Load plugin module
            plugin_instance = self._load_plugin_module(metadata, config)
            
            if plugin_instance:
                # Initialize plugin
                if plugin_instance.initialize():
                    self.loaded_plugins[plugin_id] = plugin_instance
                    metadata.status = PluginStatus.LOADED
                    metadata.load_timestamp = datetime.now(timezone.utc)
                    
                    # Start performance monitoring
                    self.performance_monitor.start_monitoring(plugin_id, plugin_instance)
                    
                    # Setup hot-reload if enabled
                    if self.hot_reload_enabled:
                        self._setup_hot_reload(plugin_id, metadata)
                    
                    self.logger.info(f"Successfully loaded plugin {plugin_id}")
                    return True
                else:
                    self.logger.error(f"Plugin {plugin_id} initialization failed")
                    metadata.status = PluginStatus.ERROR
                    return False
            else:
                self.logger.error(f"Failed to create instance of plugin {plugin_id}")
                metadata.status = PluginStatus.ERROR
                return False
                
        except Exception as e:
            self.logger.error(f"Exception loading plugin {plugin_id}: {e}")
            metadata.status = PluginStatus.ERROR
            metadata.error_message = str(e)
            return False
    
    def unload_plugin(self, plugin_id: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            plugin_id: Plugin to unload
            
        Returns:
            True if unloading successful
        """
        if plugin_id not in self.loaded_plugins:
            self.logger.warning(f"Plugin {plugin_id} not loaded")
            return True
        
        try:
            # Check for dependents
            dependents = self._get_plugin_dependents(plugin_id)
            if dependents:
                self.logger.error(f"Cannot unload {plugin_id}, required by: {dependents}")
                return False
            
            # Cleanup plugin
            plugin = self.loaded_plugins[plugin_id]
            if plugin.cleanup():
                # Stop monitoring
                self.performance_monitor.stop_monitoring(plugin_id)
                
                # Remove from loaded plugins
                del self.loaded_plugins[plugin_id]
                
                # Update metadata
                metadata = self.registry.get_plugin(plugin_id)
                if metadata:
                    metadata.status = PluginStatus.UNLOADED
                    metadata.load_timestamp = None
                
                # Cleanup hot-reload
                if plugin_id in self.file_watchers:
                    del self.file_watchers[plugin_id]
                
                self.logger.info(f"Successfully unloaded plugin {plugin_id}")
                return True
            else:
                self.logger.error(f"Plugin {plugin_id} cleanup failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Exception unloading plugin {plugin_id}: {e}")
            return False
    
    def execute_plugin(self, plugin_id: str, input_data: Any, 
                      context: Optional[Dict] = None) -> Any:
        """
        Execute a plugin.
        
        Args:
            plugin_id: Plugin to execute
            input_data: Input data for plugin
            context: Execution context
            
        Returns:
            Plugin output
        """
        if plugin_id not in self.loaded_plugins:
            raise ValueError(f"Plugin {plugin_id} not loaded")
        
        plugin = self.loaded_plugins[plugin_id]
        start_time = time.time()
        
        try:
            # Validate input
            if not plugin.validate_input(input_data):
                raise ValueError("Plugin input validation failed")
            
            # Execute with timeout and resource limits
            if self.execution_context["sandbox_enabled"]:
                result = self._execute_in_sandbox(plugin, input_data, context)
            else:
                result = plugin.execute(input_data, context)
            
            # Update metrics
            execution_time = time.time() - start_time
            plugin._update_metrics(execution_time, True)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            plugin._update_metrics(execution_time, False)
            
            self.logger.error(f"Plugin {plugin_id} execution failed: {e}")
            raise
    
    def get_plugin_status(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive plugin status."""
        metadata = self.registry.get_plugin(plugin_id)
        if not metadata:
            return None
        
        status_info = {
            "metadata": asdict(metadata),
            "is_loaded": plugin_id in self.loaded_plugins,
            "performance_metrics": None
        }
        
        if plugin_id in self.loaded_plugins:
            plugin = self.loaded_plugins[plugin_id]
            status_info["performance_metrics"] = plugin.get_performance_metrics()
            status_info["health_check"] = self._health_check_plugin(plugin_id)
        
        return status_info
    
    def get_loaded_plugins(self) -> List[str]:
        """Get list of loaded plugin IDs."""
        return list(self.loaded_plugins.keys())
    
    def reload_plugin(self, plugin_id: str) -> bool:
        """
        Reload a plugin (unload and load again).
        
        Args:
            plugin_id: Plugin to reload
            
        Returns:
            True if reload successful
        """
        if plugin_id not in self.loaded_plugins:
            return self.load_plugin(plugin_id)
        
        # Store current config
        plugin = self.loaded_plugins[plugin_id]
        current_config = plugin.config.copy()
        
        # Unload and reload
        if self.unload_plugin(plugin_id):
            return self.load_plugin(plugin_id, current_config)
        
        return False
    
    def _load_plugin_module(self, metadata: PluginMetadata, 
                          config: Optional[Dict] = None) -> Optional[ShadowBenchPlugin]:
        """Load plugin module and create instance."""
        try:
            # Construct module path
            if metadata.entry_point.startswith("http"):
                # Remote plugin - would need to download first
                self.logger.error("Remote plugin loading not implemented")
                return None
            
            # Local plugin
            module_path = Path(metadata.entry_point)
            
            if not module_path.exists():
                # Try relative to plugin directories
                for plugin_dir in self.registry.plugin_directories:
                    candidate_path = plugin_dir / metadata.entry_point
                    if candidate_path.exists():
                        module_path = candidate_path
                        break
            
            if not module_path.exists():
                self.logger.error(f"Plugin entry point not found: {metadata.entry_point}")
                return None
            
            # Load module
            spec = importlib.util.spec_from_file_location(
                f"plugin_{metadata.plugin_id}", 
                module_path
            )
            
            if not spec or not spec.loader:
                self.logger.error(f"Failed to create module spec for {module_path}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, ShadowBenchPlugin) and 
                    obj != ShadowBenchPlugin):
                    plugin_class = obj
                    break
            
            if not plugin_class:
                self.logger.error(f"No plugin class found in {module_path}")
                return None
            
            # Create plugin instance
            plugin_instance = plugin_class(metadata.plugin_id, config)
            
            return plugin_instance
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin module {metadata.entry_point}: {e}")
            return None
    
    def _execute_in_sandbox(self, plugin: ShadowBenchPlugin, 
                          input_data: Any, context: Optional[Dict] = None) -> Any:
        """Execute plugin in sandboxed environment."""
        # Simplified sandbox - in production would use more sophisticated isolation
        
        def timeout_handler():
            raise TimeoutError(f"Plugin execution exceeded {self.execution_context['max_execution_time']} seconds")
        
        # Set up timeout
        timer = threading.Timer(self.execution_context['max_execution_time'], timeout_handler)
        timer.start()
        
        try:
            result = plugin.execute(input_data, context)
            timer.cancel()
            return result
        except Exception:
            timer.cancel()
            raise
    
    def _get_plugin_dependents(self, plugin_id: str) -> List[str]:
        """Get list of plugins that depend on the given plugin."""
        dependents = []
        
        for loaded_id, metadata in [(pid, self.registry.get_plugin(pid)) 
                                   for pid in self.loaded_plugins.keys()]:
            if metadata and plugin_id in metadata.dependencies:
                dependents.append(loaded_id)
        
        return dependents
    
    def _health_check_plugin(self, plugin_id: str) -> Dict[str, Any]:
        """Perform health check on plugin."""
        if plugin_id not in self.loaded_plugins:
            return {"status": "not_loaded", "issues": ["Plugin not loaded"]}
        
        plugin = self.loaded_plugins[plugin_id]
        issues = []
        
        # Check if plugin is responsive (try a simple operation)
        try:
            # This is a simplified health check
            plugin.validate_input(None)  # Should not crash
        except Exception as e:
            issues.append(f"Plugin not responsive: {e}")
        
        # Check performance metrics
        metrics = plugin.get_performance_metrics()
        if metrics["error_count"] > metrics["total_calls"] * 0.1:  # >10% error rate
            issues.append("High error rate detected")
        
        if metrics["average_execution_time"] > 5.0:  # >5 seconds average
            issues.append("Slow execution time detected")
        
        return {
            "status": "healthy" if not issues else "issues",
            "issues": issues,
            "last_check": datetime.now(timezone.utc).isoformat()
        }
    
    def _setup_hot_reload(self, plugin_id: str, metadata: PluginMetadata):
        """Setup hot-reload monitoring for plugin."""
        # Simplified hot-reload setup - would use file system watchers in production
        self.logger.info(f"Hot-reload setup for {plugin_id} (simplified implementation)")


class PluginPerformanceMonitor:
    """Monitor plugin performance and health metrics."""
    
    def __init__(self):
        self.monitored_plugins: Dict[str, Dict[str, Any]] = {}
        self.monitoring_active = {}
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self, plugin_id: str, plugin: ShadowBenchPlugin):
        """Start monitoring a plugin."""
        self.monitored_plugins[plugin_id] = {
            "plugin": plugin,
            "start_time": time.time(),
            "metrics_history": []
        }
        self.monitoring_active[plugin_id] = True
        
        self.logger.debug(f"Started monitoring plugin {plugin_id}")
    
    def stop_monitoring(self, plugin_id: str):
        """Stop monitoring a plugin."""
        if plugin_id in self.monitored_plugins:
            del self.monitored_plugins[plugin_id]
        
        if plugin_id in self.monitoring_active:
            del self.monitoring_active[plugin_id]
        
        self.logger.debug(f"Stopped monitoring plugin {plugin_id}")
    
    def get_monitoring_report(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Get monitoring report for plugin."""
        if plugin_id not in self.monitored_plugins:
            return None
        
        plugin_data = self.monitored_plugins[plugin_id]
        plugin = plugin_data["plugin"]
        
        return {
            "plugin_id": plugin_id,
            "monitoring_duration": time.time() - plugin_data["start_time"],
            "current_metrics": plugin.get_performance_metrics(),
            "metrics_history": plugin_data["metrics_history"],
            "health_status": self._assess_health(plugin)
        }
    
    def _assess_health(self, plugin: ShadowBenchPlugin) -> str:
        """Assess plugin health based on metrics."""
        metrics = plugin.get_performance_metrics()
        
        if metrics["total_calls"] == 0:
            return "idle"
        
        error_rate = metrics["error_count"] / metrics["total_calls"]
        
        if error_rate > 0.2:  # >20% errors
            return "unhealthy"
        elif error_rate > 0.05:  # >5% errors
            return "degraded"
        else:
            return "healthy"


# Factory functions for common plugin types
def create_attack_plugin(plugin_id: str, attack_function: Callable,
                        config: Optional[Dict] = None) -> AttackGeneratorPlugin:
    """Factory function to create attack generator plugin from function."""
    
    class FunctionAttackPlugin(AttackGeneratorPlugin):
        def __init__(self, plugin_id: str, config: Optional[Dict] = None):
            super().__init__(plugin_id, config)
            self.attack_function = attack_function
        
        def initialize(self) -> bool:
            self.is_initialized = True
            return True
        
        def execute(self, input_data: Any, context: Optional[Dict] = None) -> Any:
            return self.generate_attack(input_data, context or {})
        
        def generate_attack(self, target_data: Any, attack_config: Dict[str, Any]) -> Dict[str, Any]:
            return self.attack_function(target_data, attack_config)
        
        def cleanup(self) -> bool:
            return True
        
        def get_metadata(self) -> PluginMetadata:
            return PluginMetadata(
                plugin_id=self.plugin_id,
                name=f"Function Attack Plugin {self.plugin_id}",
                version="1.0.0",
                description="Attack plugin created from function",
                author="ShadowBench",
                plugin_type=PluginType.ATTACK_GENERATOR,
                dependencies=[],
                api_version="1.0",
                entry_point="",
                configuration_schema={},
                permissions=[]
            )
    
    return FunctionAttackPlugin(plugin_id, config)


# Initialize default plugin manager instance
_default_plugin_manager = None

def get_plugin_manager(config: Optional[Dict] = None) -> PluginManager:
    """Get default plugin manager instance."""
    global _default_plugin_manager
    
    if _default_plugin_manager is None:
        _default_plugin_manager = PluginManager(config)
    
    return _default_plugin_manager
