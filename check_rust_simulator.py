import simulator_rust
import inspect

# Check what's in the simulator_rust module
print("Module name:", simulator_rust.__name__)
print("Module contents:", dir(simulator_rust))

# Try to get the signature of the Simulator class
try:
    print("\nSimulator signature:")
    print(inspect.signature(simulator_rust.Simulator.__init__))
except Exception as e:
    print(f"Error getting signature: {e}")

# Create a simple instance to check the parameters
import numpy as np
try:
    # Create a simple configuration with 2 states
    config = np.array([10, 10], dtype=np.int64)
    n_states = len(config)
    
    # Create transition matrices
    delta = np.zeros((n_states, n_states, 2, 1), dtype=np.intp)
    null_transitions = np.ones((n_states, n_states), dtype=np.uint8)
    random_transitions = np.zeros((n_states, n_states, 2), dtype=np.intp)
    random_outputs = np.zeros((1, 2), dtype=np.intp)
    transition_probabilities = np.zeros(1, dtype=np.float64)
    
    # Create the simulator without seed
    sim = simulator_rust.Simulator(
        config, 
        delta, 
        null_transitions, 
        random_transitions, 
        random_outputs, 
        transition_probabilities
    )
    print("\nCreated simulator without seed parameter")
except Exception as e:
    print(f"\nError creating simulator without seed: {e}")
