# cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, initializedcheck=False

"""
The cython module which contains the internal simulator algorithms.

This is not intended to be interacted with directly.
It is intended for the user to only interact with the class :any:`Simulation`.
"""

''' Use these commands to enable line tracing.
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
'''

from libc.math cimport log, lgamma, sqrt
from libc.stdint cimport int64_t, uint8_t, uint32_t
cimport cython
from numpy cimport npy_intp
import numpy as np
cimport numpy as np
from cpython.pycapsule cimport PyCapsule_GetPointer
from numpy.random cimport bitgen_t
from numpy.random cimport BitGenerator
from numpy.random import PCG64
from numpy.random.c_distributions cimport \
    (random_hypergeometric, random_interval, random_multinomial, random_geometric, binomial_t)
import time


cdef class Simulator:
    """Base class for the algorithm that runs the simulation.

    The configuration is stored as an array of size q, so the states are the indices 0, ..., q-1.

    Attributes:
        config: The integer array of counts representing the current configuration.
        n: The population size (sum of config).
        t: The current number of elapsed interaction steps.
        q: The total number of states (length of config).
        is_random: A boolean that is true if there are any random transitions.
        random_depth: The largest number of random outputs from any random transition.
        gen: A BitGenerator object which is the pseudorandom number generator.
        bitgen: A pointer to the BitGenerator, needed to for the numpy random C-API.
    """
    cdef public int64_t [::1] config
    cdef public int64_t n, t
    cdef npy_intp q
    cdef public npy_intp [:,:,:,] delta
    cdef public uint8_t [:,:] null_transitions
    cdef uint8_t is_random
    cdef npy_intp [:,:,:] random_transitions
    cdef npy_intp [:,:] random_outputs
    cdef double [::1] transition_probabilities
    cdef npy_intp random_depth
    cdef BitGenerator gen
    cdef bitgen_t *bitgen

    # random_outputs being typed causes an error when the protocol is not random and the array is empty.
    def __init__(self, int64_t [::1] init_array, npy_intp [:,:,:] delta, uint8_t [:,:] null_transitions,
                 npy_intp [:,:,:] random_transitions, random_outputs,
                 double [::1] transition_probabilities, seed=None):
        """Initializes the main data structures for :any:`Simulator`.

        Args:
            init_array: An integer array of counts representing the initial configuration.
            delta: A q x q x 2 array representing the transition function.
                Delta[i, j] gives contains the two output states.
            null_transitions: A q x q boolean array where entry [i, j] says if these states have a null interaction.
            random_transitions: A q x q x 2 array. Entry [i, j, 0] is the number of possible outputs if
                transition [i, j] is random, otherwise it is 0. Entry [i, j, 1] gives the starting index to find
                the outputs in the array random_outputs if it is random.
            random_outputs: An array containing all outputs of random transitions, whose indexing information
                is contained in random_transitions.
            transition_probabilities: An array containing all random transition probabilities, whose indexing matches
                random_outputs.
            seed (optional): An integer seed for the pseudorandom number generator.
        """
        self.config = init_array
        self.n = sum(init_array)
        self.q = len(init_array)
        self.delta = delta
        self.null_transitions = null_transitions
        self.is_random = np.any(random_transitions)
        if self.is_random:
            self.random_depth = 0
            for i in range(self.q):
                for j in range(self.q):
                    self.random_depth = max(self.random_depth, random_transitions[i, j, 0])
            self.random_transitions = random_transitions
            self.random_outputs = random_outputs
            self.transition_probabilities = transition_probabilities
        self.t = 0
        self.gen = PCG64(seed=seed)
        self.bitgen = <bitgen_t *> PyCapsule_GetPointer(self.gen.capsule, "BitGenerator")

    def run(self, int64_t num_steps, double max_wallclock_time = 60 * 60):
        """Base function which will be called to run the simulation for a fixed number of steps.

        Args:
            num_steps: The number of steps to run the simulation.
            max_wallclock_time: A bound in seconds on how long the simulator will run for.
        """
        pass

    def reset(self, int64_t [::1] config, int64_t t = 0):
        """Base function which will be called to reset the simulation.

        Args:
            config: The configuration array to reset to.
            t: The new value of self.t. Defaults to 0.
        """
        pass


cdef class SimulatorSequentialArray(Simulator):
    """A Simulator that sequentially chooses random agents from an array.

    Attributes:
        population: A length-n array with entries in 0, ..., q-1 giving the states of each agent.
    """
    cdef npy_intp [::1] population

    def __init__(self, *args):
        """Initializes Simulator, then creates the population array."""
        Simulator.__init__(self, *args)
        self.make_population()

    def make_population(self):
        """Creates the array self.population.

        This is an array of agent states, where the count of each state comes from :any:`Simulator.config`.
        """
        self.population = np.zeros(self.n, dtype=np.intp)
        cdef npy_intp i, j, k = 0
        for i in range(self.q):
            for j in range(self.config[i]):
                self.population[k] = i
                k += 1

    def run(self, int64_t end_step, double max_wallclock_time = 60 * 60):
        """Samples random pairs of agents and updates them until reaching end_step."""
        cdef npy_intp i, j, k, a, b
        cdef double u
        cdef double [:] ps
        cdef double end_time = time.perf_counter() + max_wallclock_time
        while self.t < end_step and time.perf_counter() < end_time:
            # random integer in [0, ... , n-1]
            i = random_interval(self.bitgen, self.n - 1)
            j = random_interval(self.bitgen, self.n - 1)
            while i == j:
                # rejection sampling to quickly get distinct pair
                j = random_interval(self.bitgen, self.n - 1)
            a, b = self.population[i], self.population[j]
            if not self.null_transitions[a, b]:
                if self.is_random and self.random_transitions[a, b, 0]:
                    k = self.random_transitions[a, b, 1]
                    # sample from a probability distribution contained in [k, k+1, ...]
                    u = self.bitgen.next_double(self.bitgen.state) - self.transition_probabilities[k]
                    while u > 0:
                        k += 1
                        u -= self.transition_probabilities[k]
                    self.population[i], self.population[j] = self.random_outputs[k]
                else:
                    self.population[i], self.population[j] = self.delta[a, b]
                self.config[a] -= 1
                self.config[b] -= 1
                self.config[self.population[i]] += 1
                self.config[self.population[j]] += 1
            self.t += 1
        return self.config

    def reset(self, int64_t [::1] config, int64_t t = 0):
        """Reset to a given configuration.

        Sets all parameters necessary to change the configuration.

        Args:
            config: The configuration array to reset to.
            t: The new value of :any:`t`. Defaults to 0.
        """
        self.config = config
        self.t = t
        self.n = sum(config)
        self.make_population()


cdef class SimulatorMultiBatch(Simulator):
    """Uses the MultiBatch algorithm to simulate O(sqrt(n)) interactions in parallel.

    The MultiBatch algorithm comes from the paper
    *Simulating Population Protocols in Subconstant Time per Interaction*
    (https://arxiv.org/abs/2005.03584).
    Beyond the methods described in the paper, this class also dynamically switches
    to Gillespie's algorithm when the number of null interactions is high.

    Attributes:
        urn: An :any:`Urn` object which stores the configuration and has methods for sampling.
        updated_counts: An additional :any:`Urn` where agents are stored that have been
            updated during a batch.
        logn: Precomputed log(n).
        batch_threshold: Minimum number of interactions that must be simulated in each
            batch. Collisions will be repeatedly sampled up until batch_threshold
            interaction steps, then all non-colliding pairs of 'delayed agents' are
            processed in parallel.
        row_sums: Array which stores sampled counts of initiator agents
            (row sums of the 'D' matrix from the paper).
        row: Array which stores the counts of responder agents for each type of
            initiator agent (one row of the 'D' matrix from the paper).
        m: Array which holds the outputs of samples from a multinomial distribution
            for batch random transitions.
        do_gillespie: A boolean determining if the we are currently doing Gillespie steps.
        silent: A boolean determining if the configuration is silent
            (all interactions are null).
        reactions: A (num_reactions) x 4 array giving a list of reactions,
            as [input input output output]
        enabled_reactions: An array holding indices of all currently applicable reactions.
        num_enabled_reactions: The number of meaningful indices in enabled_reactions.
        propensities: A num_reactions x 1 array holding the propensities of each reaction.
            The propensity of a reaction is the probability of that reaction * (n choose 2).
        reaction_probabilities: A num_reactions x 1 array giving the probability of each
            reaction, given that those two agents interact.
        gillespie_threshold: The probability of a non-null interaction must be below this
            threshold to keep doing Gillespie steps.
        coll_table: Precomputed values to speed up the function sample_coll(r, u).
        coll_table_r_values: Values of r, giving one axis of coll_table.
        coll_table_u_values: Values of u, giving the other axis of coll_table.
        num_r_values: len(coll_table_r_values), first axis of coll_table.
        num_u_values: len(coll_table_u_values), second axis of coll_table.
        r_constant: Used in definition of coll_table_r_values.
    """

    cdef public Urn urn
    cdef Urn updated_counts
    cdef double logn
    cdef int64_t batch_threshold
    cdef int64_t [::1] row_sums, row, m
    cdef public bint do_gillespie, silent
    cdef public npy_intp [:,:] reactions
    cdef public npy_intp [:] enabled_reactions
    cdef public npy_intp num_enabled_reactions
    cdef double [::1] propensities
    cdef public double [::1] reaction_probabilities
    cdef double gillespie_threshold
    cdef binomial_t _binomial  # needed to interface with the numpy multinomial C function
    cdef int64_t [:, :] coll_table
    cdef int64_t [::1] coll_table_r_values
    cdef double [::1] coll_table_u_values
    cdef npy_intp num_u_values
    cdef npy_intp num_r_values
    cdef int64_t r_constant

    def __init__(self, *args):
        """Initializes all additional data structures needed for MultiBatch Simulator."""

        Simulator.__init__(self, *args)

        self.set_n_parameters()

        self.urn = Urn.create(self.config, self.bitgen)
        self.updated_counts = Urn.create(np.zeros(self.q, dtype=np.int64), self.bitgen)
        self.row_sums = np.zeros(self.q, dtype=np.int64)
        self.row = np.zeros(self.q, dtype=np.int64)
        self.m = np.zeros(self.random_depth, dtype=np.int64)
        self.silent = False
        self.do_gillespie = False

        # enumerate reactions for gillespie
        reactions = []
        reaction_probabilities = []
        cdef npy_intp i, j, k, a, b
        cdef double p
        for i in range(self.q):
            for j in range(i+1):
                # check if interaction is symmetric
                symmetric = False
                if sorted(np.asarray(self.delta[i,j])) == sorted(np.asarray(self.delta[j, i])):
                    if self.is_random and self.random_transitions[i, j, 0] == self.random_transitions[j, i, 0] > 0:
                        a, b = self.random_transitions[i, j, 1], self.random_transitions[j, i, 1]
                        symmetric = True
                        for k in range(self.random_transitions[i, j, 0]):
                            if sorted(np.asarray(self.random_outputs[a + k])) != \
                                    sorted(np.asarray(self.random_outputs[b + k])):
                                symmetric = False
                    else:
                        symmetric = True
                if symmetric:
                    indices = [(i, j, 1.)]
                # if interaction is not symmetric, each distinct order gets added as reactions with half proability
                else:
                    indices = [(i, j, 0.5), (j, i, 0.5)]
                for a, b, p in indices:
                    if not self.null_transitions[a, b]:
                        if self.is_random and self.random_transitions[a, b, 0]:
                            for k in range(self.random_transitions[a, b, 0]):
                                output = list(self.random_outputs[self.random_transitions[a, b, 1] + k])
                                if output != [a, b]:
                                    reactions.append([a, b] + output)
                                    reaction_probabilities.append(
                                        self.transition_probabilities[self.random_transitions[a, b, 1] + k] * p)
                        else:
                            reactions.append([a, b] + list(self.delta[a, b]))
                            reaction_probabilities.append(p)
        self.reactions = np.array(reactions, dtype = np.intp)
        self.reaction_probabilities = np.array(reaction_probabilities, dtype = float)
        self.propensities = np.zeros(len(self.reactions), dtype = float)
        self.enabled_reactions = np.zeros(len(self.reactions), dtype = np.intp)
        self.get_enabled_reactions()

    def set_n_parameters(self):
        """Initialize all parameters that depend on the population size n."""
        self.logn = log(self.n)
        # theoretical optimum for batch_threshold is Theta(sqrt(n / logn) * q) agents / batch
        self.batch_threshold = int(min(sqrt(self.n / self.logn) * self.q, self.n ** 0.7))
        # first rough approximation for probability of successful reaction where we want to do gillespie
        self.gillespie_threshold = 2 / sqrt(self.n)

        # build table for precomputed coll(n, r, u) values
        self.num_r_values = int(10 * log(self.n))
        self.num_u_values = int(10 * log(self.n))
        self.r_constant = max(int(1.5 * self.batch_threshold) // ((self.num_r_values - 2) ** 2), 1)
        self.coll_table = np.zeros((self.num_r_values, self.num_u_values), dtype=np.int64)
        self.coll_table_r_values = np.array([2 + self.r_constant * (i ** 2) for i in range(self.num_r_values - 1)]
                                            + [self.n], dtype=np.int64)
        self.coll_table_u_values = np.linspace(0, 1, self.num_u_values, dtype=float)
        cdef npy_intp i, j
        for i in range(self.num_r_values):
            for j in range(self.num_u_values):
                self.coll_table[i, j] = self.sample_coll(self.coll_table_r_values[i],
                                                         self.coll_table_u_values[j], has_bounds=False)

    def reset(self, int64_t [::1] config, int64_t t = 0):
        """Reset to a given configuration.

        Sets all parameters necessary to change the configuration.

        Args:
            config: The configuration array to reset to.
            t: The new value of :any:`t`. Defaults to 0.
        """
        self.config = config
        self.urn = Urn.create(self.config, self.bitgen)
        cdef int64_t n = sum(config)
        if n != self.n:
            self.n = n
            self.set_n_parameters()
        self.t = t
        self.silent = False
        self.do_gillespie = False

    def run(self, int64_t end_step, double max_wallclock_time = 60 * 60):
        """Run the simulation for a fixed number of steps.

        Args:
            end_step: Will run until self.t = end_step.
            max_wallclock_time: A bound in seconds this will run for.
        """
        cdef double end_time = time.perf_counter() + max_wallclock_time
        while self.t < end_step and time.perf_counter() < end_time:
            if self.silent:
                self.t = end_step
                return
            elif self.do_gillespie:
                self.gillespie_step(end_step)
            else:
                self.multibatch_step(end_step)

    def run_until_silent(self, int64_t [::1] config):
        """Run the simulation until silent."""
        while not self.silent:
            if self.do_gillespie:
                self.gillespie_step()
            else:
                self.multibatch_step()

    cdef (npy_intp, npy_intp) unordered_delta(self, npy_intp a, npy_intp b):
        """Chooses sender/receiver, then applies delta to input states a, b."""
        cdef double u
        cdef npy_intp k
        cdef uint32_t coin = self.bitgen.next_uint32(self.bitgen.state) & 1
        # swap roles of a, b and swap return order by using indices coin, 1-coin
        if coin:
            b, a = a, b
        if self.is_random and self.random_transitions[a, b, 0]:
            # find the appropriate random output by linear search
            k = self.random_transitions[a, b, 1]
            u = self.bitgen.next_double(self.bitgen.state) - self.transition_probabilities[k]
            while u > 0:
                k += 1
                u -= self.transition_probabilities[k]
            return self.random_outputs[k][coin], self.random_outputs[k][1-coin]
        else:
            return self.delta[a, b, coin], self.delta[a, b, 1-coin]

    def get_enabled_reactions(self):
        """Updates :any:`enabled_reactions` and :any:`num_enabled_reactions`."""
        cdef npy_intp i, reactant_1, k
        self.num_enabled_reactions = 0
        for i in range(len(self.reactions)):
            reactant_1, reactant_2 = self.reactions[i][0], self.reactions[i][1]
            if (reactant_1 == reactant_2 and self.config[reactant_1] >= 2) or \
                    (reactant_1 != reactant_2 and self.config[reactant_1] >= 1 and self.config[reactant_2] >= 1):
                self.enabled_reactions[self.num_enabled_reactions] = i
                self.num_enabled_reactions += 1

    def gillespie_step(self, int64_t t_max = 0):
        """Samples the time until the next non-null interaction and updates.

        Args:
            t_max: Defaults to 0.
                If positive, the maximum value of :any:`t` that will be reached.
                If the sampled time is greater than t_max, then it will instead
                be set to t_max and no reaction will be performed.
                (Because of the memoryless property of the geometric, this gives a
                faithful simulation up to step t_max).
        """

        cdef npy_intp [:] r
        cdef double total_propensity = self.get_total_propensity()
        if total_propensity == 0:
            self.silent = True
            return
        cdef double n = self.n
        cdef double success_probability = total_propensity / (n * (n-1) / 2)
        cdef bint enabled_reactions_changed = False

        if success_probability > self.gillespie_threshold:
            self.do_gillespie = False
        # add a geometric number of steps, based on success probability
        new_t = self.t + random_geometric(self.bitgen, success_probability)
        self.t += random_geometric(self.bitgen, success_probability)
        # if t_max was exceeded, stop at step t_max without performing a reaction
        if self.t > t_max > 0:
            self.t = t_max
            return
        # sample the successful reaction r, currently just using linear search
        x = self.bitgen.next_double(self.bitgen.state) * total_propensity
        cdef npy_intp i = 0
        while x > 0:
            x -= self.propensities[self.enabled_reactions[i]]
            i += 1
        r = self.reactions[self.enabled_reactions[i - 1]]
        # updated with the successful reaction r
        # if any products were not already present, will updated enabled_reactions
        if self.config[r[2]] == 0 or self.config[r[3]] == 0:
            enabled_reactions_changed = True
        ## this is a bit wasteful, but want to make sure the urn data structure stays intact
        self.urn.add_to_entry(r[0], -1)
        self.urn.add_to_entry(r[1], -1)
        self.urn.add_to_entry(r[2], 1)
        self.urn.add_to_entry(r[3], 1)
        # if any reactants are now absent, will updated enabled_reactions
        if enabled_reactions_changed or self.config[r[0]] == 0 or self.config[r[1]] == 0:
            self.get_enabled_reactions()

    cpdef double get_total_propensity(self):
        """Calculates the probability the next interaction is non-null."""
        cdef npy_intp i, j
        # make sure these are all doubles, because they will be squared and could overflow int64_t
        cdef double a, b
        cdef double total_propensity = 0
        for j in range(self.num_enabled_reactions):
            i = self.enabled_reactions[j]
            a, b = self.config[self.reactions[i][0]], self.config[self.reactions[i][1]]
            if self.reactions[i][0] == self.reactions[i][1]:
                self.propensities[i] = (a * (a-1) / 2) * self.reaction_probabilities[i]
            else:
                self.propensities[i] = a * b * self.reaction_probabilities[i]
            total_propensity += self.propensities[i]
        return total_propensity

    cpdef void multibatch_step(self, int64_t t_max = 0):
        """Sample collisions to build a batch, then update the entire batch in parallel.

        See the paper for a more detailed explanation of the algorithm.
        """
        cdef int64_t num_delayed, l
        cdef double t1, t2, t3, u, r, end_step
        cdef npy_intp a, b, c, i, j, i_max, j_max, o_i, o_j


        self.updated_counts.reset()
        self.updated_counts.order = self.urn.order
        # start with count 2 of delayed agents (guaranteed for the next interaction)
        num_delayed = 2

        t1 = time.perf_counter()
        # batch will go for at least batch_threshold interactions, unless passing t_max
        end_step = self.t + self.batch_threshold
        if t_max > 0:
            end_step = min(end_step, t_max)
        while self.t + num_delayed // 2 < end_step:
            u = self.bitgen.next_double(self.bitgen.state)
            l = self.sample_coll(r = num_delayed + self.updated_counts.size,
                                 u = u, has_bounds=True)
            # add (l-1) // 2 pairs of delayed agents, the lth agent a was already picked, so has a collision
            num_delayed += 2 * ((l-1) // 2)

            # If the sampled collision happens after t_max, then include delayed agents up until t_max
            #   and do not perform the collision.
            if self.t + num_delayed // 2 >= t_max > 0:
                num_delayed = (t_max - self.t) * 2
                break

            # sample if a was a delayed or an updated agent
            u = self.bitgen.next_double(self.bitgen.state)
            r = num_delayed / (num_delayed + self.updated_counts.size)
            # true with probability num_delayed / (num_delayed + num_updated)
            if u * (num_delayed + self.updated_counts.size) <= num_delayed:
                # if a was delayed, need to first update a with its first interaction before the collision
                # c is the delayed partner that a interacted with, so add this interaction
                a = self.urn.sample_one()
                c = self.urn.sample_one()
                a, c = self.unordered_delta(a,c)
                self.t += 1
                # c is moved from delayed to updated, a is currently uncounted
                self.updated_counts.add_to_entry(c, 1)
                num_delayed -= 2
            else:
                # if a was updated, we simply sample a and remove it from updated counts
                a = self.updated_counts.sample_one()

            if l % 2 == 0:  # when l is even, the collision must with with a formally untouched agent
                b = self.urn.sample_one()
            else: # when l is odd, the collision is with the next agent, either untouched, delayed, or updated
                u = self.bitgen.next_double(self.bitgen.state)
                if u * (self.n - 1) < self.updated_counts.size:
                    # b is an updated agent, simply remove it
                    b = self.updated_counts.sample_one()
                else:
                    # we simply remove b from C is b is untouched
                    b = self.urn.sample_one()
                    # if b was delayed, we have to do the past interaction
                    if u * (self.n - 1) < self.updated_counts.size + num_delayed:
                        c = self.urn.sample_one()
                        b, c = self.unordered_delta(b,c)
                        self.t += 1
                        self.updated_counts.add_to_entry(c, 1)
                        num_delayed -= 2

            a, b = self.unordered_delta(a,b)
            self.t += 1
            self.updated_counts.add_to_entry(a, 1)
            self.updated_counts.add_to_entry(b, 1)

        t2 = time.perf_counter()
        self.do_gillespie = True  # if entire batch are null reactions, stays true and switches to gillspie
        i_max = self.urn.sample_vector(num_delayed // 2, self.row_sums)
        for i in range(i_max):
            o_i = self.urn.order[i]
            j_max = self.urn.sample_vector(self.row_sums[o_i], self.row)
            for j in range(j_max):
                o_j = self.urn.order[j]
                if self.is_random and self.random_transitions[o_i, o_j, 0]:
                    # don't switch to gillespie because we did a random transition
                    # TODO: this might not switch to gillespie soon enough in certain cases
                    self.do_gillespie = False
                    a = self.random_transitions[o_i, o_j, 0]  # better written using walrus operator
                    b = self.random_transitions[o_i, o_j, 1]
                    # updates the first a entries of m to hold a multinomial,
                    # giving the number of times for each random transition
                    self.m[:] = 0
                    random_multinomial(self.bitgen, self.row[o_j], &self.m[0], &self.transition_probabilities[b],
                                       a, &self._binomial)
                    for c in range(a):
                        self.updated_counts.add_to_entry(self.random_outputs[b+c,0], self.m[c])
                        self.updated_counts.add_to_entry(self.random_outputs[b+c,1], self.m[c])
                else:
                    if self.do_gillespie:
                        # if transition is non-null, we will set do_gillespie = False
                        self.do_gillespie = self.null_transitions[o_i, o_j]
                    # We are directly adding to updated_counts.config rather than using the function
                    #   updated_counts.add_to_entry for speed. None of the other urn features of updated_counts will
                    #   be used until it is reset in the next loop, so this is fine.
                    self.updated_counts.config[self.delta[o_i, o_j, 0]] += self.row[o_j]
                    self.updated_counts.config[self.delta[o_i, o_j, 1]] += self.row[o_j]

        self.t += num_delayed // 2
        # TODO: this is the only part scaling when the number of states (but not reached states) blows up
        self.urn.add_vector(self.updated_counts.config)

        t3 = time.perf_counter()

        # Dynamically update batch threshold, by comparing the times t2 - t1 of the collision sampling and
        #   the time t_3 - t_2 of the batch processing. Batch_threshold is adjusted to try to ensure
        #   t_2 - t_1 = t_3 - t_2
        self.batch_threshold = int(((t3 - t2) / (t2 - t1)) ** 0.1 * self.batch_threshold)
        # Keep the batch threshold within some fixed bounds.
        self.batch_threshold = min(self.batch_threshold, 2 * self.n // 3)
        self.batch_threshold = max(self.batch_threshold, 3)

        self.urn.sort()

        # update enabled_reactions if switching to gillespie
        if self.do_gillespie:
            self.get_enabled_reactions()

    cdef int64_t sample_coll(self, int64_t r, double u, bint has_bounds=True):
        """Returns a sample l ~ coll(n, r) from the collision length distribution.
        
        See Lemma 3 in the source paper. The distribution gives the number of agents needed to pick an agent twice,
        when r unique agents have already been selected.
        Inversion sampling with binary search is used, based on the formula 
            P(l > t) = (n - r)! / (n - r - t)! / (n^t).
        We sample a uniform random variable u, and find the value t such that 
            P(l > t) < U < P(l > t - 1). 
        Taking logarithms and using the lgamma function, this required formula becomes
            P(l > t) < U
             <-->
            lgamma(n - r + 1) - lgamma(n - r - t + 1) - t * log(n) < log(u).
        We will do binary search with bounds t_lo, t_hi that maintain the invariant
            P(l > t_hi) < U and P(l > t_lo) >= U.
        Once we get t_lo = t_hi - 1, we can then return t = t_hi as the output.
        
        A value of fixed outputs for u, r will be precomputed, which gives a lookup table for starting values
        of t_lo, t_hi. This function will first get called to give coll(n, r_i, u_i) for a fixed range of values 
        r_i, u_i. Then actual samples of coll(n, r, u) will find values r_i <= r < r_{i+1} and u_j <= u < u_{j+1}.
        By monotonicity in u, r, we can then set t_lo = coll(n, r_{i+i}, u_{j+1}) and t_hi = coll(n, r_i, u_j).
        
        Args:
            r: The number of agents which have already been chosen.
            u: A uniform random variable.
            has_bounds: Has the table for precomputed values of r, u already been computed?
                (This will be false while the function is being called to populate the table.)
            
        Returns:
            The number of sampled agents to get the first collision (including the collided agent).
        """
        cdef int64_t t_lo, t_mid, t_hi
        cdef npy_intp i, j
        cdef double logu, lhs
        logu = log(u)
        lhs = lgamma(self.n-r+1) - logu
        # The condition P(l < t) < U becomes
        #     lhs < lgamma(n - r - t + 1) + t * log(n)

        if has_bounds:
            # Look up bounds from coll_table.

            # For r values, we invert the definition of self.coll_table_r_values:
            #   np.array([2 + self.r_constant * (i ** 2) for i in range(self.num_r_values - 1)] + [self.n])
            i = int(sqrt((r - 2) / self.r_constant))
            i = min(i, self.num_r_values - 2)

            # for u values we similarly invert the definition: np.linspace(0, 1, num_u_values)
            j = int(u * (self.num_u_values - 1))

            # assert self.coll_table_r_values[i] <= r <= self.coll_table_r_values[i+1]
            # assert self.coll_table_u_values[j] <= u <= self.coll_table_u_values[j+1]
            t_lo = self.coll_table[i + 1, j + 1]
            t_hi = min(self.coll_table[i, j], self.n - r + 1)
        else:
            # When building the table, we start with bounds that always hold.
            if r >= self.n:
                return 1
            t_lo = 0
            t_hi = self.n - r

        # We maintain the invariant that P(l > t_lo) >= u and P(l > t_hi) < u
        # Equivalently, lhs >= lgamma(n - r - t_lo + 1) + t_lo * logn and
        #               lhs <  lgamma(n - r - t_hi + 1) + t_hi * logn
        while t_lo < t_hi - 1:
            t_mid = (t_lo + t_hi) // 2
            if lhs < lgamma(self.n - r + 1 - t_mid) + t_mid * self.logn:
                t_hi = t_mid
            else:
                t_lo = t_mid
        return t_hi


cdef class Urn:
    """Data structure for a multiset that supports fast random sampling.

    Attributes:
        config: The integer array giving the counts of the multiset.
        bitgen: Pointer to a BitGenerator, needed for numpy random C API.
        order: An integer array giving the ranking of the counts from largest to smallest.
        size: sum(config).
        length: len(config).
    """

    cdef public int64_t [::1] config
    cdef bitgen_t *bitgen
    cdef npy_intp [::1] order
    cdef int64_t size
    cdef npy_intp length

    @staticmethod
    cdef Urn create(int64_t [::1] config, bitgen_t * bitgen):
        """Initializes a new Urn object.
        
        Args:
            config: The configuration array for the urn.
            bitgen: The BitGenerator pointer. Because this is not a Python object, 
                we cannot using the typical __init__ constructor. 
                Calling this create method is a workaround.
        """
        cdef Urn urn = Urn.__new__(Urn)
        urn.config = config
        urn.bitgen = bitgen
        urn.size = sum(config)
        urn.length = len(config)
        urn.order = np.array(range(len(config)), dtype=np.intp)
        urn.sort()
        return urn

    cdef void sort(self):
        """Updates self.order.
        
        Uses insertion sort to maintain that 
            config[order[0]] >= config[order[1]] >= ... >= config[order[q]].
        This method is used to have O(q) time when order is almost correct.
        """
        cdef npy_intp i, j, k, o_i

        for i in range(1,len(self.config)):
            # See if the entry at self.order[i] needs to be moved earlier.
            # Recursively, we have ensured that order[0], ..., order[i-1] have the correct order.
            o_i = self.order[i]
            # j will be the index where self.order[i] should be inserted to.
            j = i
            while j > 0 and self.config[o_i] > self.config[self.order[j-1]]:
                j -= 1
            # Index at order[i] will get moved to order[j], and all indices order[j], ..., order[i-1] get right shifted
            # First do the right shift, moving order[i-k] for k = 1, ..., i-j
            for k in range(1, i-j+1):
                self.order[i + 1 - k] = self.order[i - k]
            self.order[j] = o_i

    cdef npy_intp sample_one(self):
        """Samples and removes one element, returning its index.
        
        Returns:
            The index of the random sample from the urn.
        """
        cdef npy_intp index, i=0
        cdef int64_t x = random_interval(self.bitgen, self.size - 1)
        while x >= 0:
            index = self.order[i]
            x -= self.config[index]
            i += 1
        self.config[index] -= 1
        self.size -= 1
        return index

    cdef void add_to_entry(self, npy_intp index, int64_t amount = 1):
        """Adds one element at index.
        
        Args:
            index: The index to add to.
            amount: The integer amount to add / subtract.
        """
        self.config[index] += amount
        self.size += amount

    cdef npy_intp sample_vector(self, int64_t n, int64_t [::1] v):
        """Samples n elements, returning them as a vector.
        
        Args:
            n: number of elements to sample
            v: the array to write the output vector in
                (this is faster than re-initializing an output array)
            
        Returns:
            nz: the number of nonzero entries
                v[self.order[i]] for i in range(nz) can then loop over only 
                    the nonzero entries of the vector
        """
        cdef int64_t init_n = n
        cdef npy_intp index, entries, i = 0
        cdef int64_t total = self.size
        cdef int64_t h
        v[:] = 0
        while n > 0 and i < self.length - 1:
            index = self.order[i]
            total -= self.config[index]
            h = random_hypergeometric(self.bitgen, self.config[index], total, n)
            v[index] = h
            n -= h
            self.size -= h
            self.config[index] -= h
            i += 1
        if n:
            v[self.order[i]] = n
            self.config[self.order[i]] -= n
            self.size -= n
            i += 1
        return i

    cdef void add_vector(self, int64_t [::1] vector):
        """Adds a vector of elements to the urn.
        
        Args:
            vector: An integer vector to add to the urn.
        """
        cdef npy_intp i = 0
        for i in range(self.length):
            self.config[i] += vector[i]
            self.size += vector[i]

    cdef void reset(self):
        """Set the counts back to zero."""
        self.config[:] = 0
        self.size = 0
