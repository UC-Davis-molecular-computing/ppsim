from ppsim import Simulation
from typing import NamedTuple
from math import log2, ceil
import simulator
import numpy as np
import protocols

import pstats, cProfile




class Agent(NamedTuple):
    value: int
    x: int
    y: int
    output: int


def initial_agent(opinion, n):
    L = 2 ** ceil(log2(n))
    if opinion.lower() == 'a':
        return Agent(value=L, x=L, y=0, output=1)
    elif opinion.lower() == 'b':
        return Agent(value=-L, x=0, y=L, output=-1)
    else:
        raise ValueError('opinion must be "a" or "b"')


def reduce(u, v):
    if u == v:
        return 0, 0
    elif u == 2 * v:
        return u - v, 0
    elif 2 * u == v:
        return 0, v - u
    else:
        return u, v


def cancel(x1, y1, x2, y2):
    x1o, y2o = reduce(x1, y2)
    x2o, y1o = reduce(x2, y1)
    return x1o, y1o, x2o, y2o


def join(x1, y1, x2, y2):
    if x1 - y1 > 0 and x2 - y2 > 0 and y1 == y2:
        y1o, y2o = y1 + y2, 0
    else:
        y1o, y2o = y1, y2
    if x1 - y1 < 0 and x2 - y2 < 0 and x1 == x2:
        x1o, x2o = x1 + x2, 0
    else:
        x1o, x2o = x1, x2
    return x1o, y1o, x2o, y2o


def split(x1, y1, x2, y2):
    if (x1 - y1 > 0 or x2 - y2 > 0) and max(x1, x2) > 1 and min(x1, x2) == 0:
        x1o = x2o = max(x1, x2) // 2
    else:
        x1o, x2o = x1, x2
    if (x1 - y1 < 0 or x2 - y2 < 0) and max(y1, y2) > 1 and min(y1, y2) == 0:
        y1o = y2o = max(y1, y2) // 2
    else:
        y1o, y2o = y1, y2
    return x1o, y1o, x2o, y2o


def normalize(x, y, v):
    x1, y1 = reduce(x, y)
    if x1 == y1 == 0:
        if v >= 0:
            output = +1
        else:
            output = -1
    else:
        if x1 > y1:
            output = +1
        else:
            output = -1
    return Agent(x=x1, y=y1, output=output, value=x1 - y1)


def split_join_majority(a, b):
    if a.x == a.y == b.x == b.y == 0:
        return a, b
    else:
        x1, y1, x2, y2 = split(*join(*cancel(a.x, a.y, b.x, b.y)))
    return normalize(x1, y1, x2 - y2), normalize(x2, y2, x1 - y1)

n = 10 ** 9
# sim = Simulation({'A': n // 2, 'B': n // 2}, protocols.approximate_majority)
# sim = Simulation({0: n // 2, 499: n // 2}, protocols.discrete_averaging)
sim = Simulation({0: n}, protocols.self_stabilizing_clock, m = 50)
# sim = Simulation({initial_agent('a',n): n // 2 + 1, initial_agent('b', n): n // 2}, split_join_majority)
cProfile.runctx("sim.run(30)", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
#
# # sim.simulator.urn.r_gap = 4
#
# from line_profiler import LineProfiler
# lp = LineProfiler()
# # lp.add_function(simulator.SimulatorMultiBatch.multibatch_step)
# # lp.add_function(simulator.SimulatorMultiBatch.run)
# # lp.add_function(simulator.DynamicAliasTable.check_to_rebuild)
# # lp.add_function(simulator.DynamicAliasTable.sample_vector)
#
# lp_wrapper = lp(sim.run)
# lp_wrapper(4)
# lp.print_stats()
#
# print(sim.simulator.sort_times)
# print(np.array(sim.simulator.sort_times).mean())
