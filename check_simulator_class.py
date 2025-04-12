import ppsim.simulator
import inspect

# Check the signature of the SimulatorSequentialArray class
print("SimulatorSequentialArray signature:")
print(inspect.signature(ppsim.simulator.SimulatorSequentialArray.__init__))

# Check the signature of the Simulator class
print("\nSimulator signature:")
print(inspect.signature(ppsim.simulator.Simulator.__init__))
