"""
This script uses the Rust implementation to run the approximate majority CRN.
"""

import sys
import time
import numpy as np

# Remove any existing imports
if 'ppsim' in sys.modules:
    del sys.modules['ppsim']

print("Importing ppsim module...")
from ppsim import species, Simulation, use_rust_simulator

# Use the Rust implementation
use_rust_simulator(True)

# Create a simple configuration for approximate majority
n = 10 ** 6  # 1 million agents
p = 0.51     # Slight bias towards A

# Define the CRN using the species syntax
a, b, u = species('A B U')
approx_majority_crn = [
    a + b >> 2 * u,  # A + B -> U + U
    a + u >> 2 * a,  # A + U -> A + A
    b + u >> 2 * b,  # B + U -> B + B
]

# Initial configuration
init_config = {a: int(p * n), b: int((1-p) * n), u: 0}

print(f"Initial configuration: A={init_config[a]}, B={init_config[b]}, U={init_config[u]}")

print("Creating Rust simulator...")
sim = Simulation(init_config=init_config, rule=approx_majority_crn, simulator_method="MultiBatch")

# Print the simulator class
print(f"Simulator class: {type(sim.simulator)}")
print(f"Simulator method: {sim._method}")

print("Running Rust simulator...")
start_time = time.time()
sim.run(run_until=20)  # Run for 20 time units
end_time = time.time()

print("Rust simulator completed!")
final_config = sim.config_dict
print(f"Final configuration: A={final_config[a]}, B={final_config[b]}, U={final_config[u]}")
print(f"Time taken: {end_time - start_time:.4f} seconds")
