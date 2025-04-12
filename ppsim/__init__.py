from ppsim.simulation import *
from ppsim.snapshot import *
from ppsim.crn import *
from ppsim.__version__ import version

# Variable to track which simulator implementation is being used
_using_rust_simulator = False

def use_rust_simulator(use_rust=True):
    """
    Switch between the Cython and Rust implementations of the simulator.
    
    Args:
        use_rust: If True, use the Rust implementation. If False, use the Cython implementation.
        
    Returns:
        The name of the implementation being used.
    """
    global _using_rust_simulator
    import sys
    import importlib
    
    # Store the current state
    was_using_rust = _using_rust_simulator
    
    # Update the state
    _using_rust_simulator = use_rust
    
    # If we're changing the implementation
    if was_using_rust != use_rust:
        # Remove the simulator module from sys.modules if it exists
        if 'ppsim.simulator' in sys.modules:
            del sys.modules['ppsim.simulator']
        
        # Import the appropriate module
        if use_rust:
            try:
                # Try to import the Rust implementation
                import simulator_rust
                # Check if SimulatorSequentialArray exists in the module
                if hasattr(simulator_rust, 'SimulatorSequentialArray'):
                    # Use the Rust implementation
                    sys.modules['ppsim.simulator'] = simulator_rust
                else:
                    # Fall back to Cython implementation
                    _using_rust_simulator = False
                    import ppsim.simulator as simulator_impl
                    return "Cython (Rust SimulatorSequentialArray not available)"
                return "Rust"
            except ImportError as e:
                # If the Rust implementation is not available, fall back to Cython
                print(f"Error importing Rust simulator: {e}")
                _using_rust_simulator = False
                import ppsim.simulator as simulator_impl
                return "Cython (Rust not available)"
        else:
            # Import the Cython implementation
            import ppsim.simulator as simulator_impl
            
            return "Cython"
    else:
        # If we're not changing the implementation, just return the current one
        return "Rust" if use_rust else "Cython"

def get_simulator_implementation():
    """
    Get the name of the simulator implementation being used.
    
    Returns:
        The name of the implementation being used.
    """
    return "Rust" if _using_rust_simulator else "Cython"
