"""
This script checks if the simulator_rust module exists and can be imported.
"""

import sys
import importlib.util

print("Python path:")
for path in sys.path:
    print(f"  {path}")

print("\nChecking if simulator_rust module exists:")
try:
    spec = importlib.util.find_spec("simulator_rust")
    if spec is not None:
        print("simulator_rust module found!")
        print(f"Location: {spec.origin}")
        
        # Try to import it
        print("\nTrying to import simulator_rust:")
        import simulator_rust
        print("Successfully imported simulator_rust")
        print(f"Module: {simulator_rust}")
        print(f"Dir: {dir(simulator_rust)}")
        
        # Check if it has a Simulator class
        if hasattr(simulator_rust, "Simulator"):
            print("\nSimulator class found in simulator_rust")
            print(f"Simulator: {simulator_rust.Simulator}")
        else:
            print("\nSimulator class NOT found in simulator_rust")
    else:
        print("simulator_rust module NOT found")
except Exception as e:
    print(f"Error: {e}")
