#!/usr/bin/env python3
"""
Minimal test to import UnifiedScorer class
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

# Test import step by step
print("Step 1: Testing basic imports...")
try:
    import logging
    import json
    import numpy as np
    from typing import Dict, List, Any, Optional, Tuple
    from dataclasses import dataclass, field
    from pathlib import Path
    print("✓ Basic imports successful")
except Exception as e:
    print(f"✗ Basic imports failed: {e}")
    sys.exit(1)

print("\nStep 2: Loading unified_scoring module...")
try:
    with open('metrics/unified_scoring.py', 'r') as f:
        code = f.read()
    
    # Execute the module code
    module_globals = {}
    exec(code, module_globals)
    
    print("✓ Module executed successfully")
    print(f"Available in module: {[k for k in module_globals.keys() if not k.startswith('__')]}")
    
    # Try to access UnifiedScorer
    if 'UnifiedScorer' in module_globals:
        UnifiedScorer = module_globals['UnifiedScorer']
        print(f"✓ UnifiedScorer found: {UnifiedScorer}")
        
        # Try to create an instance
        scorer = UnifiedScorer()
        print(f"✓ UnifiedScorer instance created: {scorer}")
    else:
        print("✗ UnifiedScorer not found in module globals")
        
except Exception as e:
    print(f"✗ Module execution failed: {e}")
    import traceback
    traceback.print_exc()
