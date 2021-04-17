"""
A module for simulating population protocols.

The main class Simulation is created with a description of the protocol and
    the initial condition, and is responsible for running the simulation.

Snapshot is a base class for snapshot objects that get are updated by
    Simulation, used to visualize the protocol during or after the simulation
    has ran.

StatePlotter is a subclass of Snapshot that gives a barplot visualizing the
    counts of all states and how they change over time.

time_trials is a convenience function used for gathering data about the
    convergence time of a protocol.
"""

import dataclasses
import time
from typing import Union, Hashable, Dict, Tuple, Callable, Optional, List

import ipywidgets as widgets
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

import simulator

# TODO: these names are not showing up in the mouseover information
State = Hashable
Output = Union[Tuple[State, State], Dict[Tuple[State, State], float]]
Rule = Union[Callable[[State, State], Output], Dict[Tuple[State, State], Output]]


# TODO: give other option for when the number of reachable states is large or unbounded
def state_enumeration(init_dist: Dict[State, int], rule: Callable[[State, State], Output], **kwargs) -> set:
    """Finds all reachable states by breadth-first search.

    Args:
        init_dist: dictionary mapping states to counts
            (states are any hashable type, commonly NamedTuple or String)
        rule: function mapping a pair of states to either a pair of states
            or to a dictionary mapping pairs of states to probabilities
        **kwargs: any additional parameters used by rule

    Returns:
        a set of all reachable states
    """
    checked_states = set()
    unchecked_states = set(init_dist.keys())
    while len(unchecked_states) > 0:
        unchecked_state = unchecked_states.pop()
        if unchecked_state not in checked_states:
            checked_states.add(unchecked_state)
        for checked_state in checked_states:
            for new_states in [rule(checked_state, unchecked_state, **kwargs),
                               rule(unchecked_state, checked_state, **kwargs)]:
                if new_states is not None:
                    if type(new_states) == dict:
                        # if the output is a distribution
                        new_states = sum(new_states.keys(), ())
                    for new_state in new_states:
                        if new_state not in checked_states and new_state not in unchecked_states:
                            unchecked_states.add(new_state)
    return checked_states


def rule_from_dict(d: Dict[Tuple[State, State], Output]):
    """Converts a dict defining a rule into a function defining a rule.

    Args:
        d: dictionary mapping pairs of states to outputs

    Returns:
        a function mapping the same pairs of states to outputs
    """
    def rule(a, b):
        if (a, b) in d:
            return d[(a, b)]
    return rule


class Snapshot:
    """"Base class for Snapshot objects.

    Attributes:
        simulation: The Simulation object that initialized and will update the Snapshot.
            This attribute gets set when the Simulation object calls add_snapshot.
        update_time: How many seconds will elapse between calls to update while
            running sim.
        time: The parallel time at the current snapshot. Changes when self.update is
            called.
        config: The configuration array at the current snapshot. Changes when
            self.update is called.
    """
    def __init__(self):
        """Init construction for the base class.

        Parameters can be passed in here, and any attributes that can be defined
        without the parent Simulation object can be instantiated here, such as
        self.update_interval.
        """
        self.simulation = None
        self.update_time = 0.1
        self.time = None
        self.config = None

    def initialize(self):
        """Method which is called once during add_snapshot.

        Any initialization that requires accessing the data in self.simulation
        should go here.
        """
        pass

    def update(self, index: Optional[int] = None):
        """Method which is called while the Simulation is running.

        Args:
            index: An optional integer index. If present, the snapshot will use the
                data from the configuration at self.sim.configs[index] and time
                self.sim.times[index]. Otherwise, the snapshot will use the current
                configuration self.sim.config_array and current time self.sim.time.
        """
        if index is None:
            self.time = self.simulation.time
            self.config = self.simulation.config_array
        else:
            self.time = self.simulation.times[index]
            self.config = self.simulation.configs[index]


class TimeUpdate(Snapshot):
    """Simple Snapshot that prints the current time in the Simulator.

    When calling Simulator.run, if there are no current Snapshots present, then
    this object will get added to provide a basic progress update.
    """
    def update(self, index: Optional[int] = None):
        super().update(index)
        print(f'\r Time: {self.time:.3f}', end='\r')


class Simulation:
    """Class to simulate a population protocol.

    Attributes:
        state_list (List[State]): A sorted list of all reachable states.
        state_dict (Dict[State, int]): Maps states to their integer index to be used
            in array representations.
        simulator (Simulator): An internal Simulator object, whose methods actually
            perform the steps of the simulation.
        configs (List[nparray[int]]): A list of all configurations that have been
            recorded during the simulation, as integer arrays.
        times (List[float]): A list of all the corresponding times of for configs,
            in units of parallel time (steps / population size n).
        column_names: Columns representing all states for pandas dataframe.
            If the State is a tuple, NamedTuple, or dataclass, this will be a
            pandas MultiIndex based on the various fields.
            Otherwise it is list of str(State) for each State.
        snapshots (List[Snapshot]): A list of Snapshot objects, which get
            periodically called during the running of the simulation to give live
            updates.
    """

    def __init__(self, init_config: Dict[State, int], rule: Rule, simulator_method: str = "MultiBatch",
                 transition_order: str = "asymmetric", **kwargs):
        """Initialize a Simulation.

        Args:
            init_config (Dict[State, int]): The starting configuration, as a
                dictionary mapping states to counts.
            rule: A representation of the transition rule. This can either be a
                dictionary, whose keys are tuples of 2 states and values are their
                outputs, or a function which takes pairs of states as input. For a
                deterministic transition function, the output is a tuple of 2 states.
                For a probabilistic transition function, the output is a dictionary
                mapping tuples of states to probabilities. Inputs that are not present
                in the dictionary, or return None from the function, are interpreted as
                null transitions that return the same pair of states as the output.
            simulator_method (str): Which Simulator method to use, either 'MultiBatch'
                or 'Sequential'.
                The MultiBatch simulator does O(sqrt(n)) interaction steps in parallel
                using batching, and is much faster for large population sizes and
                relatively small state sets.
                The Sequential represents the population as an array of agents, and
                simulates each interaction step by choosing a pair of agents to update.
                Defaults to 'MultiBatch'.
            symmetric (str): Should the rule be interpreted as being symmetric, either
                'asymmetric', 'symmetric', or 'symmetric_enforced'.
                Defaults to 'asymmetric'.
                'asymmetric': Ordering of the inputs matters, and all inputs not
                    explicitly given as assumed to be null interactions.
                'symmetric': The input pairs are interpreted as unordered. If rule(a, b)
                    returns None, while rule(b, a) has a non-null output, then the
                    output of rule(a, b) is assumed to be the same as rule(b, a).
                    If rule(a, b) and rule(b, a) are each given, there is no effect.
                    Asymmetric interactions can be explicitly included this way.
                'symmetric_enforced': The same as symmetric, except that if rule(a, b)
                    and rule(b, a) are non-null and do not give the same set of outputs,
                    a ValueError is raised.
            **kwargs: If rule is a function, other keyword function parameters are
                passed in here.
        """
        if type(rule) == dict:
            self.rule = rule_from_dict(rule)
        elif callable(rule):
            self.rule = rule
        else:
            raise TypeError("rule must be either a dict or a callable.")

        # Get a list of all reachable states, use the natsort library to put in a nice order.
        self.state_list = natsorted(list(state_enumeration(init_config, self.rule, **kwargs)),
                                    key=lambda x:x.__repr__())
        q = len(self.state_list)
        self.state_dict = {self.state_list[i]: i for i in range(q)}

        # Build the data structures necessary to instantiate the Simulator class.
        delta = np.zeros((q, q, 2), dtype=np.intp)
        null_transitions = np.zeros((q, q), dtype=bool)
        random_transitions = np.zeros((q, q, 2), dtype=np.intp)
        random_outputs = []
        transition_probabilities = []
        for i, a in enumerate(self.state_list):
            for j, b in enumerate(self.state_list):
                output = self.rule(a, b, **kwargs)
                # when output is a distribution
                if type(output) == dict:
                    s = sum(output.values())
                    assert s <= 1, "The sum of output probabilities must be <= 1."
                    # ensure probabilities sum to 1
                    if 1 - s:
                        if (a, b) in output.keys():
                            output[(a, b)] += 1 - s
                        else:
                            output[(a, b)] = 1 - s
                    if len(output) == 1:
                        # distribution only had 1 output, not actually random
                        output = next(iter(output.keys()))
                    else:
                        # add (number of outputs, index to outputs)
                        random_transitions[i, j] = (len(output), len(random_outputs))
                        for (x, y) in output.keys():
                            random_outputs.append((self.state_dict[x], self.state_dict[y]))
                        transition_probabilities.extend(list(output.values()))
                if output is None or set(output) == {a, b}:
                    null_transitions[i, j] = True
                    delta[i, j] = (i, j)
                elif type(output) == tuple:
                    delta[i, j] = (self.state_dict[output[0]], self.state_dict[output[1]])
        random_outputs = np.array(random_outputs, dtype=np.intp)
        transition_probabilities = np.array(transition_probabilities, dtype=float)

        if transition_order.lower() in ['symmetric', 'symmetric_enforced']:
            for i in range(q):
                for j in range(q):
                    # Set the output for i, j to be equal to j, i if null
                    if null_transitions[i, j]:
                        null_transitions[i, j] = null_transitions[j, i]
                        delta[i, j] = delta[j, i]
                        random_transitions[i, j] = random_transitions[j, i]
                    # If i, j and j, i are both non-null, with symmetric_enforced, check outputs are equal
                    elif transition_order.lower() == 'symmetric_enforced' and not null_transitions[j, i]:
                        if sorted(delta[i, j]) != sorted(delta[j, i]) or \
                                random_transitions[i, j, 0] != random_transitions[j, i, 0]:
                            a, b = self.state_list[i], self.state_list[j]
                            raise ValueError(f'''Asymmetric interaction:
                                            {a, b} -> {self.rule(a,b,**kwargs)}
                                            {b, a} -> {self.rule(b, a,**kwargs)}''')

        if simulator_method.lower() == 'multibatch':
            method = simulator.SimulatorMultiBatch
        elif simulator_method.lower() == 'sequential':
            method = simulator.SimulatorSequentialArray
        else:
            raise ValueError('simulator_method must be multibatch or sequential')

        self.simulator = method(self.array_from_dict(init_config), delta, null_transitions,
                                random_transitions, random_outputs, transition_probabilities)

        # Check an arbitrary state to see if it has fields.
        # This will be true for either a tuple, NamedTuple, or dataclass.
        state = next(iter(init_config.keys()))
        # Check for dataclass.
        if dataclasses.is_dataclass(state):
            field_names = [field.name for field in dataclasses.fields(state)]
            tuples = [dataclasses.astuple(state) for state in self.state_list]
        else:
            # Check for NamedTuple.
            field_names = getattr(state, '_fields', None)
            tuples = self.state_list
        # Check also for tuple.
        if field_names or isinstance(state, tuple):
            self.column_names = pd.MultiIndex.from_tuples(tuples, names=field_names)
        else:
            self.column_names = [str(i) for i in self.state_list]
        self.configs = []
        self.times = []
        self.add_config()
        # private history dataframe is initially empty, updated by the getter of property self.history
        self._history = pd.DataFrame(data=self.configs, index=pd.Index(self.times, name='time'),
                                     columns=self.column_names)
        self.snapshots = []

    def array_from_dict(self, d):
        """Convert a configuration dictionary to an array.

        Args:
            d: A dictionary mapping states to counts.

        Returns: An array giving counts of all states, in the order of
            self.state_list.
        """

        a = np.zeros(len(self.state_list), dtype=np.int64)
        for k in d.keys():
            a[self.state_dict[k]] += d[k]
        return a

    def run(self, run_until=None, recording_step=1., convergence_step=1., timer=True):
        """Runs the simulation.

        Can give a fixed amount of time to run the simulation, or a function that checks
        the configuration for convergence.

        Args:
            run_until: The stop condition. To run for a fixed amount of parallel time, give
                a numerical value. To run until a convergence criterion, give a function
                mapping a configuration (as a dictionary mapping states to counts) to a
                boolean. The run will stop when the function returns True.
                Defaults to None. If None, the simulation will run until the configuration
                is silent (all transitions are null). This only works with the multibatch
                simulator method, if another simulator method is given, then using None will
                raise a ValueError.
            recording_step: The length to run the simulator before recording each step of data,
                in units of parallel time (n steps). Defaults to 1.
            convergence_step: The length to run the simulator before checking for the stop
                condition
            timer: If True, and there are no other snapshot objects, a default TimeUpdate
                Snapshot will be created to print the current parallel time.
        """
        if len(self.snapshots) is 0 and timer is True:
            self.add_snapshot(TimeUpdate())

        # stop_condition() returns True when it is time to stop
        if run_until is None:
            if type(self.simulator) != simulator.SimulatorMultiBatch:
                raise ValueError('Running until silence only works with multibatch simulator.')

            def stop_condition():
                return self.simulator.silent
        elif type(run_until) is float or type(run_until) is int:
            end_time = self.time + run_until

            def stop_condition():
                return self.time >= end_time
        elif callable(run_until):

            def stop_condition():
                return run_until(self.config_dict)
        else:
            raise TypeError('run_until must be a float, int, function, or None.')

        # step_length is the number of interaction steps between recordings
        step_length = int(recording_step * self.simulator.n)

        # interval_length is the number of interaction steps to run for. In slow simulations, it will shrink to
        # accomodate more rapid snapshots.
        interval_length = min(step_length, int(convergence_step * self.simulator.n))

        next_step = self.simulator.t + step_length

        for snapshot in self.snapshots:
            snapshot.next_snapshot_time = time.perf_counter() + snapshot.update_time
        while stop_condition() is False:
            self.simulator.run(interval_length)
            if self.simulator.t >= next_step:
                self.add_config()
                next_step = self.simulator.t + step_length
            for snapshot in self.snapshots:
                t = time.perf_counter()
                if t >= snapshot.next_snapshot_time:
                    # if waiting for interval length took 2x the snapshot rate, cut interval length in half
                    if t >= snapshot.next_snapshot_time + snapshot.update_time:
                        interval_length = max(interval_length // 2, 1)
                    snapshot.update()
                    snapshot.next_snapshot_time = time.perf_counter() + snapshot.update_time
        # add the final configuration if running until silence
        if run_until is None:
            self.add_config()
        # final updates for all snapshots
        for snapshot in self.snapshots:
            snapshot.update()

        if len(self.snapshots) == 1 and type(self.snapshots[0]) is TimeUpdate:
            self.snapshots.pop()

    @property
    def reactions(self):
        """A string showing all non-null transitions in reaction notation.

        Each reaction is separated by \n, so that print(self.reactions) will
            display all reactions.
        """
        w = max([len(str(state)) for state in self.state_list])
        reactions = [self._reaction_string(r, p, w) for (r, p) in
                     zip(self.simulator.reactions, self.simulator.reaction_probabilities)]
        return '\n'.join(reactions)

    @property
    def enabled_reactions(self):
        """A string showing all non-null transitions that are currently enabled.

        This can only check the current configuration in self.simulator.
        Each reaction is separated by \n, so that print(self.enabled_reactions) will
            display all enabled reactions.
        """
        w = max([len(str(state)) for state in self.state_list])
        self.simulator.get_enabled_reactions()

        reactions = []
        for i in range(self.simulator.num_enabled_reactions):
            r = self.simulator.reactions[self.simulator.enabled_reactions[i]]
            p = self.simulator.reaction_probabilities[self.simulator.enabled_reactions[i]]
            reactions.append(self._reaction_string(r, p, w))
        return '\n'.join(reactions)

    def _reaction_string(self, reaction, p=1, w=1):
        """A string representation of a reaction."""

        reactants = [self.state_list[i] for i in sorted(reaction[0:2])]
        products = [self.state_list[i] for i in sorted(reaction[2:])]
        s = '{0}, {1}  -->  {2}, {3}'.format(*[str(x).rjust(w) for x in reactants + products])
        if p < 1:
            s += f'      with probability {p}'
        return s

    def reset(self, init_config: Optional[Dict[State, int]] = None):
        """Reset the Simulation.

        Args:
            init_config: The configuration to reset to. Defaults to None.
                If None, will use the old initial configuration.
        """
        if init_config is None:
            config = self.configs[0]
        else:
            config = np.zeros(len(self.state_list), dtype=np.int64)
            for k in init_config.keys():
                config[self.state_dict[k]] += init_config[k]
        self.configs = [config]
        self.times = [0]
        self._history = pd.DataFrame(columns=self.column_names)
        self.simulator.reset(config)

    def set_config(self, config):
        """Change the current configuration.

        Args:
            config: The configuration to change to. This can be a dictionary,
                mapping states to counts, or an array giving counts in the order
                of state_list.
        """
        if type(config) is dict:
            config_array = self.array_from_dict(config)
        else:
            config_array = np.array(config, dtype=np.int64)
        t = self.time
        self.simulator.reset(config_array)
        self.simulator.t = t * self.simulator.n
        self.add_config()

    @property
    def time(self):
        """The current parallel time of the simulator."""
        return self.simulator.t / self.simulator.n

    @property
    def config_dict(self):
        """The current configuration, as a dictionary mapping states to counts."""
        return {self.state_list[i]: self.simulator.config[i] for i in np.nonzero(self.simulator.config)[0]}

    @property
    def config_array(self):
        """The current configuration in the simulator, as an array of counts.

        The array is given in the same order as self.state_list. The index of state s
        is self.state_dict[s].
        """
        return np.asarray(self.simulator.config)

    @property
    def history(self):
        """A pandas dataframe containing the history of all recorded configurations."""
        h = len(self._history)
        if h < len(self.configs):
            new_history = pd.DataFrame(data=self.configs[h:], index=pd.Index(self.times[h:], name='time'),
                                       columns=self.column_names)
            self._history = pd.concat([self._history, new_history])
        return self._history

    def add_config(self):
        """Appends the current simulator configuration and time."""
        self.configs.append(np.array(self.simulator.config))
        self.times.append(self.time)

    def set_snapshot_time(self, time):
        """Updates all snapshots to the nearest recorded configuration to a specified time.

        Args:
            time (float): The parallel time to update the snapshots to.
        """
        index = np.searchsorted(self.times, time)
        for snapshot in self.snapshots:
            snapshot.update(index=index)

    def set_snapshot_index(self, index):
        """Updates all snapshots to the configuration self.configs[index].

        Args:
            index (int): The index of the configuration.
        """
        for snapshot in self.snapshots:
            snapshot.update(index=index)

    def add_snapshot(self, snap: "Snapshot"):
        """Add a new Snapshot to self.snapshots.

        Args:
            snap (Snapshot): The Snapshot object to be added.
        """
        snap.simulation = self
        snap.initialize()
        snap.update()
        self.snapshots.append(snap)

    def snapshot_slider(self, var: str = 'index'):
        """Returns a slider that updates all Snapshot objects.

        Returns a slider from the ipywidgets library.

        Args:
            var: What variable the slider uses, either 'index' or 'time'.
        """
        if var.lower() == 'index':
            return widgets.interactive(self.set_snapshot_index,
                                       index=widgets.IntSlider(min=0,
                                                               max=len(self.times) - 1,
                                                               layout=widgets.Layout(width='100%'),
                                                               step=1))
        elif var.lower() == 'time':
            return widgets.interactive(self.set_snapshot_time,
                                       time=widgets.FloatSlider(min=self.times[0],
                                                                max=self.times[-1],
                                                                layout=widgets.Layout(width='100%'),
                                                                step=0.01))
        else:
            return ValueError("var must be either 'index' or 'time'.")

    def sample_silence_time(self):
        """Starts a new trial from the initial distribution and return time until silence."""
        self.simulator.run_until_silent(np.array(self.configs[0]))
        return self.time


class StatePlotter(Snapshot):
    """Snapshot gives a barplot showing counts of states in a given configuration.

    The requires an interactive matplotlib backend to work.

    Attributes:
        fig: The matplotlib figure that is created which holds the barplot.
        ax: The matplotlib axis object that holds the barplot. Modifying properties
            of this object is the most direct way to modify the plot.
        state_map: A function mapping states to categories, which acts as a filter
            to view a subset of the states or just one field of the states.
        categories: A list which holds the set {state_map(state)} for all states
            in state_list. This gives the set of labels for the bars in the barplot.
        _matrix: A (# states)x(# categories) matrix such that for the configuration
            array self.config (indexed by states), matrix * config gives an array
            of counts of categories. Used internally for the update function.
    """
    def __init__(self, state_map=None):
        """Initializes the StatePlotter.

        Args:
            state_map: An optional function mapping states to categories.
        """
        self._matrix = None
        self.state_map = state_map

    def _add_state_map(self, state_map):
        """An internal function called to update self.categories and self.matrix."""
        self.categories = list(set([state_map(state) for state in self.simulation.state_list
                                    if state_map(state) is not None]))
        categories_dict = {j: i for i, j in enumerate(self.categories)}
        self._matrix = np.zeros((len(self.simulation.state_list), len(self.categories)), dtype=np.int64)
        for i, state in enumerate(self.simulation.state_list):
            m = state_map(state)
            if m is not None:
                self._matrix[i, categories_dict[m]] += 1

    def initialize(self):
        """Initializes the barplot.

        If self.state_map gets changed, call initialize to update the barplot to
            show the new set self.categories.
        """
        self.update_time = 0.2
        self.fig, self.ax = plt.subplots()
        if self.state_map is not None:
            self._add_state_map(self.state_map)
        else:
            self.categories = self.simulation.state_list
        self.ax = sns.barplot(x=[str(c) for c in self.categories], y=np.zeros(len(self.categories)))
        # rotate the x-axis labels if any of the label strings have more than 2 characters
        if max([len(str(c)) for c in self.categories]) > 2:
            self.ax.set_xticklabels(self.ax.get_xticklabels(), rotation=90, ha='center')
        self.ax.set_ylim(0, self.simulation.simulator.n)
        self.fig.tight_layout()

    def update(self, index=None):
        """Update the heights of all bars in the plot."""
        super().update(index)
        if self._matrix is not None:
            heights = np.matmul(self.config, self._matrix)
        else:
            heights = self.config
        for i, rect in enumerate(self.ax.patches):
            rect.set_height(heights[i])

        self.ax.set_title(f'Time {self.time}')
        self.fig.canvas.draw()


def time_trials(rule: Rule, ns: List[int], initial_conditions: Union[Callable, List],
                convergence_condition: Optional[Callable] = None, convergence_check_interval: float = 0.1,
                num_trials: int = 100, max_wallclock_time: float = 60 * 60 * 24, **kwargs):
    """Gathers data about the convergence time of a rule.

    Args:
        rule: The rule that is used to generate the Simulation.
        ns: A list of population sizes n to sample from.
            This should be in increasing order to make the time_bound work.
        initial_conditions: An initial condition is a dict mapping states to counts.
            This can either be a list of initial conditions, or a function mapping
            population size n to an initial condition of n agents.
        convergence_condition: A boolean function that takes a configuration dict
            as input and returns True if that configuration has converged.
            Defaults to None. If None, the simulation will run until silent
            (all transitions are null), and the data will be for silence time.
        convergence_check_interval: How often (in parallel time) the simulation will
            run between convergence checks. Defaults to 0.1.
            Smaller values give better resolution, but spend more time checking for
            convergence.
        num_trials: The maximum number of trials that will be done for each
            population size n, if there is sufficient time. Defaults to 100.
            If you want to ensure that you get the full num_trials samples for
            each value n, use a large value for time_bound.
        max_wallclock_time: A bound (in seconds) for how long this function will run.
            Each value n is given a time budget based on the time remaining, and
            will stop before doing num_trials runs when this time budget runs out.
            Defaults to 60 * 60 * 24 (one day).
        **kwargs: Other keyword arguments to pass into Simulation.

    Returns:
        df: A pandas dataframe giving the data from each trial, with two columns
            'n' and 'time'. A good way to visualize this dataframe is using the
            seaborn library, calling sns.lineplot(x='n', y='time', data=df).
    """
    d = {'n': [], 'time': []}
    end_time = time.perf_counter() + max_wallclock_time
    if callable(initial_conditions):
        initial_conditions = [initial_conditions(n) for n in ns]
    for i in tqdm(range(len(ns))):
        sim = Simulation(initial_conditions[i], rule)
        t = time.perf_counter()
        time_limit = t + (end_time - t) / (len(ns) - i)
        j = 0
        while j < num_trials and time.perf_counter() < time_limit:
            j += 1
            sim.reset(initial_conditions[i])
            sim.run(convergence_condition, convergence_step=convergence_check_interval, timer=False)
            d['n'].append(ns[i])
            d['time'].append(sim.time)

    return pd.DataFrame(data=d)

