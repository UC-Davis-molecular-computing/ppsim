"""
This script directly uses the ppsim.simulator module to run the approximate majority CRN.
"""

import sys
import time
import numpy as np

# Remove any existing imports
if 'ppsim' in sys.modules:
    del sys.modules['ppsim']
if 'ppsim.simulator' in sys.modules:
    del sys.modules['ppsim.simulator']

print("Importing ppsim.simulator module...")
import ppsim.simulator

# Create a simple configuration for approximate majority
# State 0 = A, State 1 = B, State 2 = U
n = 10 ** 6  # 1 million agents
p = 0.51     # Slight bias towards A
config = np.zeros(3, dtype=np.int64)
config[0] = int(p * n)  # A
config[1] = int((1-p) * n)  # B
config[2] = 0  # U
n_states = len(config)

print(f"Initial configuration: A={config[0]}, B={config[1]}, U={config[2]}")

# Create transition matrices for approximate majority
# A + B -> U + U
# A + U -> A + A
# B + U -> B + B

# Delta matrix: shape (n_states, n_states, 2)
delta = np.zeros((n_states, n_states, 2), dtype=np.intp)

# A + B -> U + U
delta[0, 1, 0] = 2  # First agent becomes U
delta[0, 1, 1] = 2  # Second agent becomes U
delta[1, 0, 0] = 2  # First agent becomes U
delta[1, 0, 1] = 2  # Second agent becomes U

# A + U -> A + A
delta[0, 2, 0] = 0  # First agent stays A
delta[0, 2, 1] = 0  # Second agent becomes A
delta[2, 0, 0] = 0  # First agent becomes A
delta[2, 0, 1] = 0  # Second agent stays A

# B + U -> B + B
delta[1, 2, 0] = 1  # First agent stays B
delta[1, 2, 1] = 1  # Second agent becomes B
delta[2, 1, 0] = 1  # First agent becomes B
delta[2, 1, 1] = 1  # Second agent stays B

# Null transitions: shape (n_states, n_states)
null_transitions = np.ones((n_states, n_states), dtype=np.uint8)
null_transitions[0, 1] = 0  # Allow interaction between A and B
null_transitions[1, 0] = 0  # Allow interaction between B and A
null_transitions[0, 2] = 0  # Allow interaction between A and U
null_transitions[2, 0] = 0  # Allow interaction between U and A
null_transitions[1, 2] = 0  # Allow interaction between B and U
null_transitions[2, 1] = 0  # Allow interaction between U and B

# No random transitions
random_transitions = np.zeros((n_states, n_states, 2), dtype=np.intp)
random_outputs = np.zeros((1, 2), dtype=np.intp)  # Dummy array, not used
transition_probabilities = np.zeros(1, dtype=np.float64)  # Dummy array, not used

print("Creating Cython simulator...")
sim = ppsim.simulator.SimulatorMultiBatch(
    config, 
    delta, 
    null_transitions, 
    random_transitions, 
    random_outputs, 
    transition_probabilities
)

print("Running Cython simulator...")
start_time = time.time()
sim.run(int(n * 20))  # Run for 20 time units
end_time = time.time()

print("Cython simulator completed!")
print(f"Final configuration: A={sim.config[0]}, B={sim.config[1]}, U={sim.config[2]}")
print(f"Time taken: {end_time - start_time:.4f} seconds")
