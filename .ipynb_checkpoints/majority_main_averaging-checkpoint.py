from ppsim import ppsim
from typing import NamedTuple, Optional
import numpy as np


# TODO: figure out best way to implement symmetric reactions in the function code

class MajorityAgent(NamedTuple):
    # input: str = 'A'
    # output: Optional[str] = 'A'
    role: str = 'Main'
    minute: Optional[int] = None
    hour: Optional[int] = None
    exponent: Optional[int] = None
    bias: Optional[int] = None


def make_agent(input):
    if input == 'A':
        return MajorityAgent(bias=1, exponent=0)
    if input == 'B':
        return MajorityAgent(bias=-1, exponent=0)
    if input == 'C':
        return MajorityAgent(role='Clock', minute=0)


def majority_main_averaging(a: MajorityAgent, b: MajorityAgent, L: int, k: int, p: float = 1):
    new_a = a._asdict()
    new_b = b._asdict()
    if a.role == b.role == 'Clock':
        if a.minute == b.minute < L * k:
            # clock drip reaction
            new_a['minute'] += 1
            return {(MajorityAgent(**new_a), MajorityAgent(**new_b)): p}
        else:
            # clock epidemic reaction
            new_a['minute'] = new_b['minute'] = max(a.minute, b.minute)

    # clock update reaction
    if a.role == 'Main' and a.bias == 0 and b.role == 'Clock':
        new_a['hour'] = max(a.hour, b.minute // k)
    if b.role == 'Main' and b.bias == 0 and a.role == 'Clock':
        new_b['hour'] = max(b.hour, a.minute // k)

    if a.role == b.role == 'Main':
        # cancel reaction
        if {-1, 1}.issubset({a.bias, b.bias}) and a.exponent == b.exponent:
            new_a['bias'] = new_b['bias'] = 0
            new_a['exponent'] = new_b['exponent'] = None
            new_a['hour'] = new_b['hour'] = -a.exponent
        # split reaction
        if a.bias == 0 and b.bias != 0 and abs(a.hour) > abs(b.exponent):
            new_a['bias'] = b.bias
            new_a['hour'] = None
            new_a['exponent'] = new_b['exponent'] = b.exponent - 1
        if b.bias == 0 and a.bias != 0 and abs(b.hour) > abs(a.exponent):
            new_b['bias'] = a.bias
            new_b['hour'] = None
            new_a['exponent'] = new_b['exponent'] = a.exponent - 1
    return MajorityAgent(**new_a), MajorityAgent(**new_b)


def get_one_field(df, field):
    return df.transpose().groupby(level=field).sum().transpose()

n = 10 ** 7
init_dist = {make_agent('A'): n // 4 + 1, make_agent('B'): n // 4 - 1, make_agent('C'): n // 2}
sim = ppsim.Simulation(init_dist, majority_main_averaging,
                       L=int(np.log2(n))+1, k=3, p=1/4)

# lp = LineProfiler()
# # lp.add_function(sim.simulator.sample_coll)
# # lp.add_function(sim.simulator.get_enabled_reactions)
# # lp.add_function(sim.simulator.multibatch_step)
# # lp.add_function(sim.simulator.gillespie_step)
# lp.add_function(sim.simulator.run)
# # lp.add_function(sim.simulator.urn.sample_one)
# # lp.add_function(sim.simulator.urn.sample_vector)
# # lp.add_function(sim.simulator.urn.add_vector)
#
# lp_wrapper = lp(sim.run_until_silent)
# lp_wrapper()
#
# # lp_wrapper = lp(sim.run)
# # lp_wrapper(200)
#
# lp.print_stats()