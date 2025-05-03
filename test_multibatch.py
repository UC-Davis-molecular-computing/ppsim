"""
Basic test script for the ppsim package.
"""

import numpy as np
from ppsim import SimulatorMultiBatch

    

def test_simulator_multi_batch():
    """Test the SimulatorMultiBatch class."""
    init_config = np.array([1,1,1], dtype=np.int64)
    delta = np.array([
        [[0,0], [1,1], [2,2]],
        [[1,1], [1,1], [2,2]],
        [[2,2], [2,2], [2,2]],
    ], dtype=np.int64)
    null_transitions = np.zeros((3,3), dtype=np.bool)
    random_transitions = np.zeros((3,3,2), dtype=np.int64)
    random_outputs = np.zeros((1, 2), dtype=np.int64)
    transition_probabilities = np.ones(1, dtype=np.float64)
    sim = SimulatorMultiBatch(
        init_config,
        delta,
        null_transitions,
        random_transitions,
        random_outputs,
        transition_probabilities,
        0,
    )
    
    # Run the simulator
    sim.run(100, 10)

if __name__ == "__main__":
    test_simulator_multi_batch()
