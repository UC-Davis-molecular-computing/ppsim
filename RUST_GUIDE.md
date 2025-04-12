# Using ppsim with Rust and Cython Backends

This guide explains how to use the ppsim package with both Cython and Rust backends, how to compile the code locally, and how to switch between implementations for performance comparison.

## Prerequisites

- Python 3.7+
- Rust (install from [rustup.rs](https://rustup.rs/))
- Maturin (`pip install maturin`)
- NumPy (`pip install numpy`)

## Installation

### Local Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/UC-Davis-molecular-computing/ppsim.git
   cd ppsim
   ```

2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

3. Build the Rust extension:
   ```bash
   maturin develop
   ```

## Switching Between Cython and Rust Implementations

The ppsim package provides a function to switch between Cython and Rust implementations:

```python
from ppsim import use_rust_simulator, get_simulator_implementation

# Switch to Rust implementation
use_rust_simulator(True)
print(f"Using {get_simulator_implementation()} implementation")

# Switch to Cython implementation
use_rust_simulator(False)
print(f"Using {get_simulator_implementation()} implementation")
```

## Basic Usage

Here's a simple example of using ppsim with both implementations:

```python
import numpy as np
from ppsim import use_rust_simulator

# Choose which implementation to use
use_rust_simulator(True)  # True for Rust, False for Cython

# Create a simple configuration with 2 states
config = np.array([10, 10], dtype=np.int64)  # 10 agents in state 0, 10 in state 1
n_states = len(config)

# Import the appropriate simulator module
if use_rust_simulator(True) == "Rust":
    import simulator_rust as simulator
    
    # Create transition matrices for Rust implementation
    # Delta matrix: shape (n_states, n_states, 2, 1)
    delta = np.zeros((n_states, n_states, 2, 1), dtype=np.intp)
    
    # Simple transition rule: state 0 + state 1 -> state 0 + state 0
    delta[0, 1, 0, 0] = 0  # First agent becomes state 0
    delta[0, 1, 1, 0] = 0  # Second agent becomes state 0
else:
    import ppsim.simulator as simulator
    
    # Create transition matrices for Cython implementation
    # Delta matrix: shape (n_states, n_states, 2)
    delta = np.zeros((n_states, n_states, 2), dtype=np.intp)
    
    # Simple transition rule: state 0 + state 1 -> state 0 + state 0
    delta[0, 1, 0] = 0  # First agent becomes state 0
    delta[0, 1, 1] = 0  # Second agent becomes state 0

# Null transitions: shape (n_states, n_states)
# 1 means no interaction, 0 means interaction happens
null_transitions = np.ones((n_states, n_states), dtype=np.uint8)
null_transitions[0, 1] = 0  # Allow interaction between state 0 and state 1
null_transitions[1, 0] = 0  # Allow interaction between state 1 and state 0

# No random transitions
random_transitions = np.zeros((n_states, n_states, 2), dtype=np.intp)
random_outputs = np.zeros((1, 2), dtype=np.intp)  # Dummy array, not used
transition_probabilities = np.zeros(1, dtype=np.float64)  # Dummy array, not used

# Create the simulator
sim = simulator.Simulator(
    config, 
    delta, 
    null_transitions, 
    random_transitions, 
    random_outputs, 
    transition_probabilities,
    seed=42
)

# Run the simulation for 100 steps
result = sim.run(100, 3600.0)  # 3600.0 seconds (1 hour) as max_wallclock_time
```

## Performance Comparison

You can use the provided `test_simulator_implementations.py` script to compare the performance of the Cython and Rust implementations:

```bash
python test_simulator_implementations.py
```

This script will run the same simulation with both implementations and report the time taken by each.

## Implementation Details

### Cython Implementation

The Cython implementation provides several simulator classes:
- `Simulator`: Base class for the algorithm that runs the simulation
- `SimulatorSequentialArray`: A Simulator that sequentially chooses random agents from an array
- `SimulatorMultiBatch`: Uses the MultiBatch algorithm to simulate O(sqrt(n)) interactions in parallel

### Rust Implementation

The Rust implementation provides:
- `Simulator`: Base class for the algorithm that runs the simulation
- `Urn`: Data structure for a multiset that supports fast random sampling

## Rebuilding the Rust Extension

If you make changes to the Rust code, you need to rebuild the extension:

```bash
maturin develop
```

## Troubleshooting

### Import Errors

If you get an error like `No module named 'simulator_rust'`, make sure you have built the Rust extension with `maturin develop`.

### Compilation Errors

If you get compilation errors when building the Rust extension, check:
1. That you have Rust installed and it's in your PATH
2. That you have the required dependencies installed
3. That the Rust code is compatible with the current version of PyO3 and numpy-rust

### Performance Issues

If one implementation is significantly slower than the other, check:
1. That you're using the correct implementation
2. That you're using the appropriate simulator class for your use case
3. That you're not hitting any edge cases that might affect performance
