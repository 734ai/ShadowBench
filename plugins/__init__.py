"""
Dynamic Plugin System for ShadowBench
Provides extensible architecture with runtime loading and REST API integration.
"""

from .plugin_manager import (
    PluginManager,
    PluginRegistry,
    ShadowBenchPlugin,
    AttackGeneratorPlugin,
    MetricCalculatorPlugin,
    ModelWrapperPlugin,
    PluginMetadata,
    PluginType,
    PluginStatus,
    PluginPerformanceMonitor,
    create_attack_plugin,
    get_plugin_manager
)

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime, timezone


class PluginAPIServer:
    """
    REST API server for plugin management and execution.
    
    Provides HTTP endpoints for:
    - Plugin discovery and management
    - Runtime plugin execution
    - Performance monitoring
    - Configuration management
    """
    
    def __init__(self, plugin_manager: PluginManager, host: str = "localhost", port: int = 8080):
        self.plugin_manager = plugin_manager
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__)
        
        # API statistics
        self.api_stats = {
            "requests_total": 0,
            "requests_by_endpoint": {},
            "errors_total": 0,
            "start_time": datetime.now(timezone.utc)
        }
    
    async def create_app(self):
        """Create the web application with routes."""
        from aiohttp import web
        
        app = web.Application()
        
        # Plugin management endpoints
        app.router.add_get('/api/v1/plugins', self.list_plugins)
        app.router.add_get('/api/v1/plugins/{plugin_id}', self.get_plugin_info)
        app.router.add_post('/api/v1/plugins/{plugin_id}/load', self.load_plugin)
        app.router.add_post('/api/v1/plugins/{plugin_id}/unload', self.unload_plugin)
        app.router.add_post('/api/v1/plugins/{plugin_id}/reload', self.reload_plugin)
        app.router.add_post('/api/v1/plugins/{plugin_id}/execute', self.execute_plugin)
        
        # Discovery endpoints
        app.router.add_post('/api/v1/plugins/discover', self.discover_plugins)
        app.router.add_get('/api/v1/plugins/types', self.list_plugin_types)
        
        # Monitoring endpoints
        app.router.add_get('/api/v1/plugins/{plugin_id}/status', self.get_plugin_status)
        app.router.add_get('/api/v1/plugins/{plugin_id}/metrics', self.get_plugin_metrics)
        app.router.add_get('/api/v1/system/stats', self.get_system_stats)
        
        # Configuration endpoints
        app.router.add_get('/api/v1/plugins/{plugin_id}/config', self.get_plugin_config)
        app.router.add_put('/api/v1/plugins/{plugin_id}/config', self.update_plugin_config)
        
        # Health check
        app.router.add_get('/api/v1/health', self.health_check)
        
        # Add middleware
        app.middlewares.append(self.logging_middleware)
        app.middlewares.append(self.error_handling_middleware)
        
        return app
    
    async def start_server(self):
        """Start the API server."""
        from aiohttp import web
        
        app = await self.create_app()
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        self.logger.info(f"Plugin API server started on {self.host}:{self.port}")
        
        return runner
    
    @web.middleware
    async def logging_middleware(self, request, handler):
        """Middleware for request logging."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            response = await handler(request)
            
            # Update statistics
            self.api_stats["requests_total"] += 1
            endpoint = f"{request.method} {request.path}"
            self.api_stats["requests_by_endpoint"][endpoint] = (
                self.api_stats["requests_by_endpoint"].get(endpoint, 0) + 1
            )
            
            # Log request
            duration = asyncio.get_event_loop().time() - start_time
            self.logger.info(
                f"{request.method} {request.path} - {response.status} - {duration:.3f}s"
            )
            
            return response
            
        except Exception as e:
            self.api_stats["errors_total"] += 1
            self.logger.error(f"Request failed: {e}")
            raise
    
    @web.middleware
    async def error_handling_middleware(self, request, handler):
        """Middleware for error handling."""
        from aiohttp import web
        
        try:
            return await handler(request)
        except Exception as e:
            self.logger.error(f"Unhandled error in {request.path}: {e}")
            
            return web.json_response({
                "error": "Internal server error",
                "message": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, status=500)
    
    async def list_plugins(self, request):
        """List all available plugins."""
        from aiohttp import web
        
        try:
            plugins = self.plugin_manager.registry.plugins
            plugin_type_filter = request.query.get('type')
            status_filter = request.query.get('status')
            
            # Apply filters
            filtered_plugins = []
            for plugin in plugins.values():
                if plugin_type_filter and plugin.plugin_type.value != plugin_type_filter:
                    continue
                if status_filter and plugin.status.value != status_filter:
                    continue
                
                plugin_info = {
                    "plugin_id": plugin.plugin_id,
                    "name": plugin.name,
                    "version": plugin.version,
                    "type": plugin.plugin_type.value,
                    "status": plugin.status.value,
                    "description": plugin.description,
                    "author": plugin.author,
                    "is_loaded": plugin.plugin_id in self.plugin_manager.loaded_plugins
                }
                filtered_plugins.append(plugin_info)
            
            return web.json_response({
                "plugins": filtered_plugins,
                "total_count": len(filtered_plugins),
                "filters_applied": {
                    "type": plugin_type_filter,
                    "status": status_filter
                }
            })
            
        except Exception as e:
            return web.json_response({
                "error": "Failed to list plugins",
                "message": str(e)
            }, status=500)
    
    async def get_plugin_info(self, request):
        """Get detailed information about a specific plugin."""
        from aiohttp import web
        
        plugin_id = request.match_info['plugin_id']
        
        try:
            status_info = self.plugin_manager.get_plugin_status(plugin_id)
            
            if not status_info:
                return web.json_response({
                    "error": "Plugin not found",
                    "plugin_id": plugin_id
                }, status=404)
            
            return web.json_response(status_info)
            
        except Exception as e:
            return web.json_response({
                "error": "Failed to get plugin info",
                "message": str(e)
            }, status=500)
    
    async def load_plugin(self, request):
        """Load a plugin."""
        from aiohttp import web
        
        plugin_id = request.match_info['plugin_id']
        
        try:
            # Parse configuration from request body
            config = {}
            if request.content_type == 'application/json':
                body = await request.json()
                config = body.get('config', {})
            
            success = self.plugin_manager.load_plugin(plugin_id, config)
            
            if success:
                return web.json_response({
                    "message": f"Plugin {plugin_id} loaded successfully",
                    "plugin_id": plugin_id,
                    "status": "loaded"
                })
            else:
                return web.json_response({
                    "error": f"Failed to load plugin {plugin_id}",
                    "plugin_id": plugin_id
                }, status=400)
                
        except Exception as e:
            return web.json_response({
                "error": "Plugin loading failed",
                "message": str(e)
            }, status=500)
    
    async def unload_plugin(self, request):
        """Unload a plugin."""
        from aiohttp import web
        
        plugin_id = request.match_info['plugin_id']
        
        try:
            success = self.plugin_manager.unload_plugin(plugin_id)
            
            if success:
                return web.json_response({
                    "message": f"Plugin {plugin_id} unloaded successfully",
                    "plugin_id": plugin_id,
                    "status": "unloaded"
                })
            else:
                return web.json_response({
                    "error": f"Failed to unload plugin {plugin_id}",
                    "plugin_id": plugin_id
                }, status=400)
                
        except Exception as e:
            return web.json_response({
                "error": "Plugin unloading failed",
                "message": str(e)
            }, status=500)
    
    async def reload_plugin(self, request):
        """Reload a plugin."""
        from aiohttp import web
        
        plugin_id = request.match_info['plugin_id']
        
        try:
            success = self.plugin_manager.reload_plugin(plugin_id)
            
            if success:
                return web.json_response({
                    "message": f"Plugin {plugin_id} reloaded successfully",
                    "plugin_id": plugin_id,
                    "status": "reloaded"
                })
            else:
                return web.json_response({
                    "error": f"Failed to reload plugin {plugin_id}",
                    "plugin_id": plugin_id
                }, status=400)
                
        except Exception as e:
            return web.json_response({
                "error": "Plugin reloading failed",
                "message": str(e)
            }, status=500)
    
    async def execute_plugin(self, request):
        """Execute a plugin."""
        from aiohttp import web
        
        plugin_id = request.match_info['plugin_id']
        
        try:
            # Parse request body
            if request.content_type != 'application/json':
                return web.json_response({
                    "error": "Request must be JSON",
                    "content_type_received": request.content_type
                }, status=400)
            
            body = await request.json()
            input_data = body.get('input_data')
            context = body.get('context', {})
            
            # Execute plugin
            result = self.plugin_manager.execute_plugin(plugin_id, input_data, context)
            
            return web.json_response({
                "result": result,
                "plugin_id": plugin_id,
                "execution_timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except ValueError as e:
            return web.json_response({
                "error": "Plugin execution failed",
                "message": str(e)
            }, status=400)
        except Exception as e:
            return web.json_response({
                "error": "Plugin execution error",
                "message": str(e)
            }, status=500)
    
    async def discover_plugins(self, request):
        """Discover plugins."""
        from aiohttp import web
        
        try:
            # Parse options
            force_refresh = False
            if request.content_type == 'application/json':
                body = await request.json()
                force_refresh = body.get('force_refresh', False)
            
            plugins = self.plugin_manager.discover_plugins()
            
            plugin_summaries = []
            for plugin in plugins:
                plugin_summaries.append({
                    "plugin_id": plugin.plugin_id,
                    "name": plugin.name,
                    "version": plugin.version,
                    "type": plugin.plugin_type.value,
                    "author": plugin.author,
                    "description": plugin.description
                })
            
            return web.json_response({
                "discovered_plugins": plugin_summaries,
                "total_discovered": len(plugin_summaries),
                "discovery_timestamp": datetime.now(timezone.utc).isoformat(),
                "force_refresh": force_refresh
            })
            
        except Exception as e:
            return web.json_response({
                "error": "Plugin discovery failed",
                "message": str(e)
            }, status=500)
    
    async def list_plugin_types(self, request):
        """List available plugin types."""
        from aiohttp import web
        
        plugin_types = [ptype.value for ptype in PluginType]
        
        return web.json_response({
            "plugin_types": plugin_types,
            "total_types": len(plugin_types)
        })
    
    async def get_plugin_status(self, request):
        """Get plugin status."""
        from aiohttp import web
        
        plugin_id = request.match_info['plugin_id']
        
        try:
            status_info = self.plugin_manager.get_plugin_status(plugin_id)
            
            if not status_info:
                return web.json_response({
                    "error": "Plugin not found",
                    "plugin_id": plugin_id
                }, status=404)
            
            return web.json_response(status_info)
            
        except Exception as e:
            return web.json_response({
                "error": "Failed to get plugin status",
                "message": str(e)
            }, status=500)
    
    async def get_plugin_metrics(self, request):
        """Get plugin performance metrics."""
        from aiohttp import web
        
        plugin_id = request.match_info['plugin_id']
        
        try:
            if plugin_id not in self.plugin_manager.loaded_plugins:
                return web.json_response({
                    "error": "Plugin not loaded",
                    "plugin_id": plugin_id
                }, status=404)
            
            plugin = self.plugin_manager.loaded_plugins[plugin_id]
            metrics = plugin.get_performance_metrics()
            
            # Get monitoring report if available
            monitoring_report = self.plugin_manager.performance_monitor.get_monitoring_report(plugin_id)
            
            response_data = {
                "plugin_id": plugin_id,
                "metrics": metrics,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            if monitoring_report:
                response_data["monitoring_report"] = monitoring_report
            
            return web.json_response(response_data)
            
        except Exception as e:
            return web.json_response({
                "error": "Failed to get plugin metrics",
                "message": str(e)
            }, status=500)
    
    async def get_system_stats(self, request):
        """Get system-wide statistics."""
        from aiohttp import web
        
        try:
            loaded_plugins = self.plugin_manager.get_loaded_plugins()
            total_plugins = len(self.plugin_manager.registry.plugins)
            
            # Calculate uptime
            uptime_seconds = (datetime.now(timezone.utc) - self.api_stats["start_time"]).total_seconds()
            
            stats = {
                "system": {
                    "uptime_seconds": uptime_seconds,
                    "total_plugins_discovered": total_plugins,
                    "loaded_plugins_count": len(loaded_plugins),
                    "loaded_plugins": loaded_plugins
                },
                "api": {
                    "total_requests": self.api_stats["requests_total"],
                    "total_errors": self.api_stats["errors_total"],
                    "requests_by_endpoint": self.api_stats["requests_by_endpoint"],
                    "error_rate": (
                        self.api_stats["errors_total"] / max(self.api_stats["requests_total"], 1)
                    )
                },
                "plugin_types": {
                    ptype.value: len(self.plugin_manager.registry.get_plugins_by_type(ptype))
                    for ptype in PluginType
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            return web.json_response(stats)
            
        except Exception as e:
            return web.json_response({
                "error": "Failed to get system stats",
                "message": str(e)
            }, status=500)
    
    async def get_plugin_config(self, request):
        """Get plugin configuration."""
        from aiohttp import web
        
        plugin_id = request.match_info['plugin_id']
        
        try:
            if plugin_id not in self.plugin_manager.loaded_plugins:
                return web.json_response({
                    "error": "Plugin not loaded",
                    "plugin_id": plugin_id
                }, status=404)
            
            plugin = self.plugin_manager.loaded_plugins[plugin_id]
            
            return web.json_response({
                "plugin_id": plugin_id,
                "config": plugin.config,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            return web.json_response({
                "error": "Failed to get plugin config",
                "message": str(e)
            }, status=500)
    
    async def update_plugin_config(self, request):
        """Update plugin configuration."""
        from aiohttp import web
        
        plugin_id = request.match_info['plugin_id']
        
        try:
            if plugin_id not in self.plugin_manager.loaded_plugins:
                return web.json_response({
                    "error": "Plugin not loaded",
                    "plugin_id": plugin_id
                }, status=404)
            
            # Parse new configuration
            if request.content_type != 'application/json':
                return web.json_response({
                    "error": "Request must be JSON"
                }, status=400)
            
            body = await request.json()
            new_config = body.get('config', {})
            
            # Update plugin configuration
            plugin = self.plugin_manager.loaded_plugins[plugin_id]
            plugin.config.update(new_config)
            
            return web.json_response({
                "message": f"Configuration updated for plugin {plugin_id}",
                "plugin_id": plugin_id,
                "new_config": plugin.config,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            return web.json_response({
                "error": "Failed to update plugin config",
                "message": str(e)
            }, status=500)
    
    async def health_check(self, request):
        """Health check endpoint."""
        from aiohttp import web
        
        return web.json_response({
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "ShadowBench Plugin API",
            "version": "1.0.0"
        })


class ShadowBenchPluginSystem:
    """
    Complete plugin system integration for ShadowBench.
    
    Combines plugin management with API server and provides
    high-level interface for plugin operations.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize plugin manager
        self.plugin_manager = PluginManager(self.config.get('plugin_manager', {}))
        
        # Initialize API server if enabled
        self.api_server = None
        if self.config.get('enable_api_server', True):
            api_config = self.config.get('api_server', {})
            self.api_server = PluginAPIServer(
                self.plugin_manager,
                host=api_config.get('host', 'localhost'),
                port=api_config.get('port', 8080)
            )
        
        self.logger.info("ShadowBench Plugin System initialized")
    
    async def start(self):
        """Start the plugin system."""
        # Discover plugins
        self.logger.info("Discovering plugins...")
        plugins = self.plugin_manager.discover_plugins()
        self.logger.info(f"Discovered {len(plugins)} plugins")
        
        # Load core plugins if specified
        core_plugins = self.config.get('core_plugins', [])
        for plugin_id in core_plugins:
            self.logger.info(f"Loading core plugin: {plugin_id}")
            if not self.plugin_manager.load_plugin(plugin_id):
                self.logger.error(f"Failed to load core plugin: {plugin_id}")
        
        # Start API server if enabled
        if self.api_server:
            await self.api_server.start_server()
        
        self.logger.info("Plugin system started successfully")
    
    def get_plugin_manager(self) -> PluginManager:
        """Get the plugin manager instance."""
        return self.plugin_manager
    
    def get_api_server(self) -> Optional[PluginAPIServer]:
        """Get the API server instance."""
        return self.api_server


# Factory function
def create_plugin_system(config: Optional[Dict] = None) -> ShadowBenchPluginSystem:
    """Create a configured plugin system."""
    return ShadowBenchPluginSystem(config)


# Utility functions for common plugin operations
def register_function_as_plugin(plugin_id: str, function: callable, 
                               plugin_type: PluginType = PluginType.ATTACK_GENERATOR,
                               config: Optional[Dict] = None) -> bool:
    """
    Register a simple function as a plugin.
    
    Args:
        plugin_id: Unique plugin identifier
        function: Function to wrap as plugin
        plugin_type: Type of plugin
        config: Plugin configuration
        
    Returns:
        True if registration successful
    """
    try:
        plugin_manager = get_plugin_manager()
        
        if plugin_type == PluginType.ATTACK_GENERATOR:
            plugin = create_attack_plugin(plugin_id, function, config)
        else:
            # For other types, would need additional factory functions
            raise ValueError(f"Function registration not supported for type: {plugin_type}")
        
        # Manually add to loaded plugins (bypassing file-based discovery)
        plugin_manager.loaded_plugins[plugin_id] = plugin
        
        return True
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to register function as plugin: {e}")
        return False


__all__ = [
    'PluginManager',
    'PluginRegistry', 
    'ShadowBenchPlugin',
    'AttackGeneratorPlugin',
    'MetricCalculatorPlugin',
    'ModelWrapperPlugin',
    'PluginMetadata',
    'PluginType',
    'PluginStatus',
    'PluginPerformanceMonitor',
    'PluginAPIServer',
    'ShadowBenchPluginSystem',
    'create_attack_plugin',
    'get_plugin_manager',
    'create_plugin_system',
    'register_function_as_plugin'
]
