# ppsim Python package

The `ppsim` package is used for simulating population protocols. The package and further example notebooks can be found on [Github](https://github.com/UC-Davis-molecular-computing/ppsim).

The core of the simulator uses a [batching algorithm](https://arxiv.org/abs/2005.03584) which gives significant asymptotic gains for protocols with relatively small reachable state sets. The package is designed to be run in a Python notebook, to concisely describe complex protocols, efficiently simulate their dynamics, and provide helpful visualization of the simulation.

## Installation

The package can be installed with `pip` via


```python
pip install ppsim
```

The most important part of the package is the `Simulation` class, which is responsible for parsing a protocol, performing the simulation, and giving data about the simulation.


```python
from ppsim import Simulation
```

## Example protocol

A state can be any hashable Python object. The simplest way to describe a protocol is a dictionary mapping pairs of input states to pairs of output states.
For example, here is a description of the classic 3-state [approximate majority protocol](http://www.cs.yale.edu/homes/aspnes/papers/approximate-majority-journal.pdf). There are two initial states `A` and `B`, and the protocol converges with high probability to the majority state with the help of a third "undecided" state `U`.


```python
a, b, u = 'A', 'B', 'U'
approximate_majority = {
    (a,b): (u,u),
    (a,u): (a,a),
    (b,u): (b,b)
}
```

## Example Simulation

To instantiate a `Simulation`, we must specify a protocol along with an initial condition, which is a dictionary mapping states to counts. Let's simulate approximate majority with in a population of one billion agents with a slight majority of `A` agents.


```python
n = 10 ** 9
init_config = {a: 0.501 * n, b: 0.499 * n}
sim = Simulation(init_config, approximate_majority)
```

Now let's run this simulation for `10` units of parallel time (`10 * n` interactions). We will record the configuration every `0.1` units of time.


```python
sim.run(10, 0.1)
```

     Time: 10.000
    

The `Simulation` class can display all these configurations in a `pandas` dataframe in the attribute `history`.


```python
sim.history
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>U</th>
    </tr>
    <tr>
      <th>time (n interactions)</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>501000000</td>
      <td>499000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>459457762</td>
      <td>457439751</td>
      <td>83102487</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>430276789</td>
      <td>428217565</td>
      <td>141505646</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>409027376</td>
      <td>406898254</td>
      <td>184074370</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>393162729</td>
      <td>390949934</td>
      <td>215887337</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9.7</th>
      <td>771074143</td>
      <td>55357812</td>
      <td>173568045</td>
    </tr>
    <tr>
      <th>9.8</th>
      <td>789103074</td>
      <td>48973925</td>
      <td>161923001</td>
    </tr>
    <tr>
      <th>9.9</th>
      <td>806667929</td>
      <td>43076383</td>
      <td>150255688</td>
    </tr>
    <tr>
      <th>10.0</th>
      <td>823641388</td>
      <td>37668547</td>
      <td>138690065</td>
    </tr>
    <tr>
      <th>10.0</th>
      <td>823641389</td>
      <td>37668547</td>
      <td>138690064</td>
    </tr>
  </tbody>
</table>
<p>102 rows x 3 columns</p>
</div>




```python
p = sim.history.plot()
```


    
![png](README_files/README_12_0.png)
    


Without specifying an end time, `run` will run the simulation until the configuration is silent (all interactions are null). In this case, that will be when the protocol reaches a silent majority consensus configuration.


```python
sim.run()
p = sim.history.plot()
```

     Time: 21.000
    


    
![png](README_files/README_14_1.png)
    


Note that by default, `Simulation` assumes that input pair `(b,a)` will have the same transition as `(a,b)`, so order doesn't matter, with the default setting `transition_order = 'symmetric'`.
Thus we have the exact same protocol as if we had spent more time explicitly specifying


```python
approximate_majority_symmetric = {
    (a,b): (u,u), (b,a): (u,u),
    (a,u): (a,a), (u,a): (a,a),
    (b,u): (b,b), (u,b): (b,b)
}
```

If we intentionally meant for these pairwise transitions to only happen in this specified order, we can declare that. We see in this case that it has the same behavior, but just runs twice as slow because now every interaction must happen in a specified order.


```python
sim = Simulation(init_config, approximate_majority, transition_order='asymmetric')
print(sim.reactions)
sim.run()
p = sim.history.plot()
```

    A, B  -->  U, U      with probability 0.5
    A, U  -->  A, A      with probability 0.5
    B, U  -->  B, B      with probability 0.5
     Time: 44.000
    


    
![png](README_files/README_18_1.png)
    


A key result about this protocol is it converges in expected O(log n) time, which surprisingly is very nontrivial to prove. We can use this package to very quickly gather some convincing data that the convergence really is O(log n) time, with the function `time_trials`.


```python
from ppsim import time_trials
import numpy as np

ns = [int(n) for n in np.geomspace(10, 10 ** 8, 20)]
def initial_condition(n):
    return {'A': n // 2, 'B': n // 2}
df = time_trials(approximate_majority, ns, initial_condition, num_trials=100, max_wallclock_time = 30)
df
```



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>2.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>2.7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>3.8</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1492</th>
      <td>42813323</td>
      <td>23.8</td>
    </tr>
    <tr>
      <th>1493</th>
      <td>100000000</td>
      <td>28.1</td>
    </tr>
    <tr>
      <th>1494</th>
      <td>100000000</td>
      <td>25.2</td>
    </tr>
    <tr>
      <th>1495</th>
      <td>100000000</td>
      <td>25.1</td>
    </tr>
    <tr>
      <th>1496</th>
      <td>100000000</td>
      <td>24.6</td>
    </tr>
  </tbody>
</table>
<p>1497 rows x 2 columns</p>
</div>



This dataframe collected time from up to 100 trials for each population size n across a many orders of magnitude, limited by the budget of 30 seconds of wallclock time that we gave it.
We can now use the `seaborn` library to get a convincing plot of the data.


```python
import seaborn as sns
lp = sns.lineplot(x='n', y='time', data=df)
lp.set_xscale('log')
```


    
![png](README_files/README_22_0.png)
    


## Larger state protocol

For more complicated protocols, it would be very tedious to use this dictionary format. Instead we can give an arbitrary Python function which takes a pair of states as input (along with possible other protocol parameters) and returns a pair of states as output (or if we wanted a randomized transition, it would output a dictionary which maps pairs of states to probabilities).

As a quick example, let's take a look at the discrete averaging dynamics, as analyzed [here](https://arxiv.org/abs/1808.05389) and [here](https://hal-cnrs.archives-ouvertes.fr/hal-02473856/file/main_JAP.pdf), which have been a key subroutine used in counting and majority protocols.


```python
from math import ceil, floor

def discrete_averaging(a, b):
    avg = (a + b) / 2
    return floor(avg), ceil(avg)

n = 10 ** 6
sim = Simulation({0: n // 2, 50: n // 2}, discrete_averaging)
```

We did not need to explicitly describe the state set. Upon initialization, `Simulation` used breadth first search to find all states reachable from the initial configuration.


```python
print(sim.state_list)
```

    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    

This enumeration will call the function `rule` we give it O(q^2) times, where q is the number of reachable states. This preprocessing step also builds an internal representation of the transition function, so it will not need to continue calling `rule`. Thus we don't need to worry too much about our code for `rule` being efficient.

Rather than the dictionary format used to input the configuration, internally `Simulation` represents the configuration as an array of counts, where the ordering of the indices is given by `state_list`.


```python
sim.config_dict
```




    {0: 500000, 50: 500000}




```python
sim.config_array
```




    array([500000,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,
                0,      0, 500000], dtype=int64)



A key result about these discrete averaging dynamics is that they converge in O(log n) time to at most 3 consecutive values. It could take longer to reach the ultimate silent configuration with only 2 consecutive values, so if we wanted to check for the faster convergence condition, we could use a function that checks for the condition. This function takes a configuration dictionary (mapping states to counts) as input and returns `True` if the convergence criterion has been met.


```python
def three_consecutive_values(config):
    states = config.keys()
    return max(states) - min(states) <= 2
```

Now we can run until this condition is met (or also use `time_trials` as above to generate statistics about this convergence time).


```python
sim.run(three_consecutive_values, 0.1)
sim.history
```

     Time: 14.300
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>41</th>
      <th>42</th>
      <th>43</th>
      <th>44</th>
      <th>45</th>
      <th>46</th>
      <th>47</th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
    </tr>
    <tr>
      <th>time (n interactions)</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>500000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>500000</td>
    </tr>
    <tr>
      <th>0.1</th>
      <td>450122</td>
      <td>1</td>
      <td>1</td>
      <td>16</td>
      <td>2</td>
      <td>0</td>
      <td>364</td>
      <td>127</td>
      <td>0</td>
      <td>18</td>
      <td>...</td>
      <td>23</td>
      <td>0</td>
      <td>116</td>
      <td>344</td>
      <td>4</td>
      <td>1</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>450204</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>401218</td>
      <td>11</td>
      <td>18</td>
      <td>242</td>
      <td>48</td>
      <td>17</td>
      <td>2059</td>
      <td>692</td>
      <td>26</td>
      <td>211</td>
      <td>...</td>
      <td>236</td>
      <td>25</td>
      <td>697</td>
      <td>2053</td>
      <td>22</td>
      <td>37</td>
      <td>180</td>
      <td>2</td>
      <td>3</td>
      <td>401462</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>354315</td>
      <td>40</td>
      <td>63</td>
      <td>696</td>
      <td>147</td>
      <td>72</td>
      <td>5015</td>
      <td>1722</td>
      <td>151</td>
      <td>759</td>
      <td>...</td>
      <td>706</td>
      <td>97</td>
      <td>1725</td>
      <td>4952</td>
      <td>76</td>
      <td>163</td>
      <td>717</td>
      <td>43</td>
      <td>32</td>
      <td>354744</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>309976</td>
      <td>109</td>
      <td>135</td>
      <td>1527</td>
      <td>382</td>
      <td>247</td>
      <td>8439</td>
      <td>2994</td>
      <td>404</td>
      <td>1714</td>
      <td>...</td>
      <td>1673</td>
      <td>380</td>
      <td>2934</td>
      <td>8292</td>
      <td>249</td>
      <td>414</td>
      <td>1588</td>
      <td>140</td>
      <td>108</td>
      <td>310440</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>13.9</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14.0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14.1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14.2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14.3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>144 rows x 51 columns</p>
</div>



With a much larger number of states, the `history` dataframe is more unwieldly, so trying to directly call `history.plot()` would be very messy and not very useful.
Instead we will bring in a `Snapshot` object that makes a bar plot with the counts of each state, and lets us visualize the way the distribution evolves over time.

For this `StatePlotter` object to work as intended, we need to be using an interactive matplotlib backend, such as `%matplotlib widget` or `%matplotlib qt`. It is recommended to use `%matplotlib widget`, which uses the package [ipympl](https://github.com/matplotlib/ipympl), and to run the notebook with [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/).

Note these interactive plots are not displayed in a static notebook. Also note that some common IPython environments such as [Google Colab](https://colab.research.google.com/) do not support any of the interactive matplotlib backends, which prevents the interactive `Snapshot` objects `StatePlotter` and `HistoryPlotter` from working correctly.


```python
# Requires ipympl package, can be installed from pip with pip install ipympl
%matplotlib widget

from ppsim import StatePlotter
sp = StatePlotter(update_time=1)
sim.add_snapshot(sp)
sim.snapshot_slider('time')
```


![gif](https://raw.githubusercontent.com/UC-Davis-molecular-computing/population-protocols-python-package/main/README_files/barplot1.gif)

To better visualize small count states, let's change `yscale` to `symlog`.


```python
sp.ax.set_yscale('symlog')
```

![gif](https://raw.githubusercontent.com/UC-Davis-molecular-computing/population-protocols-python-package/main/README_files/barplot2.gif)

If we run the `Simulation` while this `Snapshot` has already been created, it will update while the simulation runs. Because the population average was exactly 25, the ultimate silent configuration will have every agent in state 50, but it will take a a very long time to reach, as we must wait for pairwise interactions between dwindling counts of states 24 and 26. We can check that this reaction is now the only possible non-null interaction.


```python
print(sim.enabled_reactions)
```

    24, 26  -->  25, 25
    

As a result, the probability of a non-null interaction will grow very small, upon which the simulator will switch to the Gillespie algorithm. This allows it to relatively quickly run all the way until silence, which we can confirm takes a very long amount of parallel time.


```python
# In order to see a Snapshot update live while the simulation is running, the command sim.add_snapshot() must be called in a previous cell
# After the Snapshot is already displayed, calling sim.run() will update it in real time
# The parameter Snapshot.update_time controls how often (in seconds) the Snapshot will get updated

# Setting history_interval to be a function of time t that shrinks, to not record too many configurations over a long time scale
sim.run(history_interval=lambda t: 10 ** len(str(int(t))) / 100)
```

Since the timescale of the whole simulation is now very long, we should have the slider range across recorded indices rather than parallel time.


```python
display(sp.fig.canvas)
sim.snapshot_slider('index')
```


![gif](https://raw.githubusercontent.com/UC-Davis-molecular-computing/population-protocols-python-package/main/README_files/barplot3.gif)

For more examples see https://github.com/UC-Davis-molecular-computing/population-protocols-python-package/tree/main/examples/

