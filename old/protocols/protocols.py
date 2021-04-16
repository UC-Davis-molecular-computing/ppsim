from collections import namedtuple
from typing import NamedTuple, Optional
from math import ceil, floor


class AgentDLE(NamedTuple):
    leader: str = 'L'
    minute: object = 0 # no clear type, with domain 0, ..., m-1, N
    flip: int = 0


def dimmed_leader_election(a: AgentDLE, b:AgentDLE, m: int):
    middle_minutes = range(m // 3, 2 * m // 3 + 1)
    # outputs agents with fields (a_l, a_m, a_f), (b_l, b_m, b_f)
    a_l, b_l = a.leader, b.leader

    if a.leader == b.leader == 'L':
        b_l = 'F'  # drop out by fratricide

    if a.minute != 'N' and b.minute != 'N': # neither are neutral
        # neighbor epidemic reactions
        if (a.minute + 1) % m == b.minute:
            a_m = b_m = b.minute
        elif (b.minute + 1) % m == a.minute:
            a_m = b_m = a.minute
        elif a.minute == b.minute:
            # a drips if a leader is present
            if 'L' in [a.leader, b.leader]:
                 # both drip ahead, so a/b can be used as coin flip
                a_m = b_m = (a.minute + 1) % m
            else:
                a_m = b_m = a.minute
        # |a.minute - b.minute| > 1, clipping reaction
        else:
            a_m = b_m = 'N'
    # recovery epidemic reactions
    elif a.minute != 'N' and b.minute == 'N':
        a_m = b_m = a.minute
    elif b.minute != 'N' and a.minute == 'N':
        a_m = b_m = b.minute
    elif a.minute == b.minute == 'N':
        a_m = b_m = 'N'

    # leaders set flip when entering minute 0, by sender / receiver bit
    if a_l == 'L' and a.minute != 0 and a_m == 0:
        a_f = 1
    else:
        a_f = a.flip
    if b_l == 'L' and b.minute != 0 and b_m == 0:
        b_f = 0
    else:
        b_f = b.flip

    # dimmed leaders become followers when entering minute 0
    if a_l == 'D' and a.minute != 0 and a_m == 0:
        a_l = 'F'
    if b_l == 'D' and b.minute != 0 and b_m == 0:
        b_l = 'F'

    # followers propagate heads by epidemic during middle minutes
    if a_l == 'F' and a_m in middle_minutes:
        a_f = max(a.flip, b.flip)
    if b_l == 'F' and b_m in middle_minutes:
        b_f = max(a.flip, b.flip)

    # followers set flip back to 0 outside the middle minutes
    if a_l == 'F' and a_m not in middle_minutes:
        a_f = 0
    if b_l == 'F' and b_m not in middle_minutes:
        b_f = 0

    # leaders who flip tails become dim after seeing a heads during the middle minutes
    if a_l == 'L' and a_m in middle_minutes and a_f == 0 and b_f == 1:
        a_l = 'D'
        a_f = 1
    if b_l == 'L' and b_m in middle_minutes and b_f == 0 and a_f == 1:
        b_l = 'D'
        b_f = 1

    # dimmed leaders have flip = 0
    if a_l == 'D':
        a_f = 0
    if b_l == 'D':
        b_f = 0

    return AgentDLE(a_l, a_m, a_f), AgentDLE(b_l, b_m, b_f)


def approximate_majority(a, b):
    if {'A', 'B'}.issubset({a, b}):
        return 'U', 'U'

    if {'A', 'U'}.issubset({a, b}):
        return 'A', 'A'

    if {'B', 'U'}.issubset({a, b}):
        return 'B', 'B'


def weighted_approximate_majority(a, b, p_a=1, p_b=1):
    if {'A', 'B'}.issubset({a, b}):
        return 'U', 'U'

    if {'A', 'U'}.issubset({a, b}):
        return {('A', 'A'): p_a}

    if {'B', 'U'}.issubset({a, b}):
        return {('B', 'B'): p_b}


class AgentEM(NamedTuple):
    output: str = 'A'
    active: bool = True

    def __str__(self):
        if self.active:
            return self.output
        else:
            return self.output.lower()


def six_state_exact_majority(a, b):
    ag = AgentEM
    if a.active and b.active and {'A', 'B'}.issubset({a.output, b.output}):
        return ag('T', True), ag('T', True)
    if a.active and not b.active:
        return ag(a.output, a.active), ag(a.output, b.active)
    if not a.active and b.active:
        return ag(b.output, a.active), ag(b.output, b.active)
    if a.active and a.output in ['A', 'B'] and b.output == 'T':
        return ag(a.output, a.active), ag(a.output, False)
    if b.active and b.output in ['A', 'B'] and a.output == 'T':
        return ag(b.output, False), ag(b.output, b.active)


def four_state_exact_majority(a, b):
    if {'A', 'B'}.issubset({a, b}):
        return 'a', 'b'
    if (a, b) == ('a', 'b'):
        return 'a', 'a'
    if (a, b) == ('b', 'a'):
        return 'b', 'b'
    if a.isupper() and b.islower():
        return a, a.lower()
    if a.islower() and b.isupper():
        return b.lower(), b


def discrete_averaging(a, b):
    avg = (a + b) / 2
    return floor(avg), ceil(avg)


def self_stabilizing_clock(a, b, m, p=1):
    # states a, b in (0, ..., m-1, 'N')
    if a in range(m) and b in range(m):
        # drip reaction
        if a == b:
            return {((a + 1) % m, b): p}
        # epidemic reaction
        elif (a + 1) % m == b:
            return b, b
        elif (b + 1) % m == a:
            return a, a
        # clipping reaction
        else:
            return 'N', 'N'
    # recovery epidemic reactions
    elif a == 'N':
        return b, b
    elif b == 'N':
        return a, a
