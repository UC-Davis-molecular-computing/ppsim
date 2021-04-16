import ppsim
from ppsim import simulator
import protocols
from line_profiler import LineProfiler

def time_test():
    n = 10 ** 7
    sim = ppsim.Simulation({0: n // 2, 499: n // 2}, protocols.discrete_averaging)
    sim.run(3)

# sim = population-protocols-python-package.Simulation({protocols.AgentEM('A', True): n // 2, protocols.AgentEM('B', True): n // 2},
#                       protocols.six_state_exact_majority)
# sim = population-protocols-python-package.Simulation({'A': n // 2 + 1, 'B': n // 2}, protocols.four_state_exact_majority)


lp = LineProfiler()
# lp.add_function(simulator.SimulatorMultiBatch.sample_coll)
# lp.add_function(simulator.SimulatorMultiBatch.multibatch_step)
# lp.add_function(sim.simulator.gillespie_step)
# lp.add_function(simulator.SimulatorMultiBatch.run)
# lp.add_function(simulator.DynamicAliasTable.sample_one)
# lp.add_function(simulator.DynamicAliasTable.sample_vector)
# lp.add_function(simulator.DynamicAliasTable.add_vector)
# lp.add_function(simulator.DynamicAliasTable.make_table)
# lp.add_function(simulator.DynamicAliasTable.check_to_rebuild)
# lp.add_function(simulator.DynamicAliasTable.add_to_entry)

# lp_wrapper = lp(sim.run_until_silent)
lp_wrapper = lp(time_test)
lp_wrapper()
lp.print_stats()

