"""
A module for :any:`Snapshot` objects used to visualize the protocol during or after
the simulation has run.

:any:`Snapshot` is a base class for snapshot objects that get are updated by :any:`Simulation`.

:any:`Plotter` is a subclass of :any:`Snapshot` that creates a matplotlib figure and axis.
It also gives the option for a state_map function which maps states to the categories which
will show up in the plot.

:any:`StatePlotter` is a subclass of :any:`Plotter` that creates a barplot of the counts
in categories.

:any:`HistoryPlotter` is a subclass of :any:`Plotter` that creates a lineplot of the counts
in categories over time.
"""

from typing import Optional, Callable, Hashable, Any

from natsort import natsorted
import numpy as np
import pandas as pd  # type: ignore
from tqdm import tqdm

State = Hashable


class Snapshot:
    """Base class for snapshot objects.

    Attributes:
        simulation: The :any:`Simulation` object that initialized and will update the
            :any:`Snapshot`.
            This attribute gets set when the :any:`Simulation` object calls
            :any:`add_snapshot`.
        update_time: How many seconds will elapse between calls to update while
            in the :any:`Simulation.run` method of :any:`simulation`.
        time: The time at the current snapshot. Changes when :any:`Snapshot.update` is called.
        config: The configuration array at the current snapshot. Changes when
            :any:`Snapshot.update` is called.
    """

    def __init__(self) -> None:
        """Init constructor for the base class.

        Parameters can be passed in here, and any attributes that can be defined
        without the parent :any:`Simulation` object can be instantiated here, such as
        :any:`update_time`.
        """
        self.simulation = None
        self.update_time = 0.1
        self.time = None
        self.config = None

    def initialize(self) -> None:
        """Method which is called once during :any:`add_snapshot`.

        Any initialization that requires accessing the data in :any:`simulation`
        should go here.
        """
        if self.simulation is None:
            raise ValueError('self.simulation is None, cannot call self.initialize until using sim.add_snapshot')

    def update(self, index: Optional[int] = None) -> None:
        """Method which is called while :any:`Snapshot.simulation` is running.

        Args:
            index: An optional integer index. If present, the snapshot will use the
                data from configuration :any:`configs` ``[index]`` and time
                :any:`times` ``[index]``. Otherwise, the snapshot will use the current
                configuration :any:`config_array` and current time.
        """
        if self.simulation is None:
            raise ValueError('self.simulation is None, cannot call self.update until using sim.add_snapshot')
        if index is not None:
            self.time = self.simulation.times[index]
            self.config = self.simulation.configs[index]
        else:
            self.time = self.simulation.time
            self.config = self.simulation.config_array


class TimeUpdate(Snapshot):
    """Simple :any:`Snapshot` that prints the current time in the :any:`Simulation`.

    When calling :any:`Simulation.run`, if :any:`snapshots` is empty, then
    this object will get added to provide a basic progress update.
    """
    def __init__(self, time_bound: Optional[float] = None, update_time: float = 0.2) -> None:
        self.pbar = tqdm(total=time_bound, position=0, leave=False, unit=' time simulated')
        self.update_time = update_time

    def initialize(self) -> None:
        self.start_time = self.simulation.time

    def update(self, index: Optional[int] = None) -> None:
        super().update(index)
        new_n = round(self.time - self.start_time, 3)
        self.pbar.update(new_n - self.pbar.n)


class Plotter(Snapshot):
    """Base class for a :any:`Snapshot` which will make a plot.

    Gives the option to map states to categories, for an easy way to visualize
    relevant subsets of the states rather than the whole state set.
    These require an interactive matplotlib backend to work.

    Attributes:
        fig: The matplotlib figure that is created.
        ax: The matplotlib axis object that is created. Modifying properties
            of this object is the most direct way to modify the plot.
        yscale: The scale used for the yaxis, passed into ax.set_yscale.
        state_map: A function mapping states to categories, which acts as a filter
            to view a subset of the states or just one field of the states.
        categories: A list which holds the set ``{state_map(state)}`` for all states
            in :any:`state_list`.
        sort_by:
        _matrix: A (# states)x(# categories) matrix such that for the configuration
            array (indexed by states), ``matrix * config`` gives an array
            of counts of categories. Used internally to get counts of categories.
    """

    def __init__(self, state_map: Optional[Callable[[State], Any]]=None, update_time=0.5, yscale='linear',
                 sort_by: str = 'categories') -> None:
        """Initializes the :any:`Plotter`.

        Args:
            state_map: An optional function mapping states to categories.
            yscale: The scale used for the yaxis, passed into ax.set_yscale.
                Defaults to 'linear'.
        """
        self._matrix = None
        self.state_map = state_map
        self.update_time = update_time
        self.yscale = yscale
        self.sort_by = sort_by

    def _add_state_map(self, state_map):
        """An internal function called to update :any:`categories` and `_matrix`."""
        self.categories = []

        for state in self.simulation.state_list:
            if state_map(state) is not None and state_map(state) not in self.categories:
                self.categories.append(state_map(state))
        self.categories = natsorted(self.categories, key=lambda x: repr(x))

        categories_dict = {j: i for i, j in enumerate(self.categories)}
        self._matrix = np.zeros((len(self.simulation.state_list), len(self.categories)), dtype=np.int64)
        for i, state in enumerate(self.simulation.state_list):
            m = state_map(state)
            if m is not None:
                self._matrix[i, categories_dict[m]] += 1

    def initialize(self) -> None:
        """Initializes the plotter by creating a fig and ax."""
        # Only do matplotlib import when necessary
        super().initialize()
        from matplotlib import pyplot as plt
        self.fig, self.ax = plt.subplots()
        if self.state_map is not None:
            self._add_state_map(self.state_map)
        else:
            self.categories = self.simulation.state_list


class StatePlotter(Plotter):
    """:any:`Plotter` which produces a barplot of counts."""

    def initialize(self) -> None:
        """Initializes the barplot.

        If :any:`state_map` gets changed, call :any:`initialize` to update the barplot to
            show the new set :any:`categories`.
        """
        super().initialize()
        import seaborn as sns
        self.ax = sns.barplot(x=[str(c) for c in self.categories], y=np.zeros(len(self.categories)))
        # rotate the x-axis labels if any of the label strings have more than 2 characters
        if max([len(str(c)) for c in self.categories]) > 2:
            for tick in self.ax.get_xticklabels():
                tick.set_rotation(90)
        self.ax.set_yscale(self.yscale)
        if self.yscale in ['symlog', 'log']:
            self.ax.set_ylim(0, 2 * self.simulation.simulator.n)
        else:
            self.ax.set_ylim(0, self.simulation.simulator.n)

    def update(self, index: Optional[int] = None) -> None:
        """Update the heights of all bars in the plot."""
        super().update(index)
        if self._matrix is not None:
            heights = np.matmul(self.config, self._matrix)
        else:
            heights = self.config
        for i, rect in enumerate(self.ax.patches):
            rect.set_height(heights[i])

        self.ax.set_title(f'Time {self.time: .3f}')
        self.fig.tight_layout()
        self.fig.canvas.draw()


class HistoryPlotter(Plotter):
    """Plotter which produces a lineplot of counts over time."""

    def update(self, index: Optional[int] = None) -> None:
        """Make a new history plot."""
        super().update(index)
        self.ax.clear()
        if self._matrix is not None:
            df = pd.DataFrame(data=np.matmul(self.simulation.history.to_numpy(), self._matrix),
                              columns=self.categories,
                              index=self.simulation.history.index)
        else:
            df = self.simulation.history
        df.plot(ax=self.ax)

        self.ax.set_yscale(self.yscale)
        if self.yscale in ['symlog', 'log']:
            self.ax.set_ylim(0, 2 * self.simulation.simulator.n)
        else:
            self.ax.set_ylim(0, 1.1 * self.simulation.simulator.n)

        # rotate the x labels if they are time units
        if self.simulation.time_units:
            for tick in self.ax.get_xticklabels():
                tick.set_rotation(45)
        self.fig.tight_layout()
        self.fig.canvas.draw()
