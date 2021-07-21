# ppsim Python package

The `ppsim` package is used for simulating population protocols. The package and further example notebooks can be found on [Github](https://github.com/UC-Davis-molecular-computing/ppsim).


If you find ppsim useful in a scientific project, please cite its associated paper:

> <ins>ppsim: A software package for efficiently simulating and visualizing population protocols</ins>.  
  David Doty and Eric Severson.  
  CMSB 2021: *Proceedings of the 19th International Conference on Computational Methods in Systems Biology*  
  [ [paper](http://arxiv.org/abs/2105.04702) | [BibTeX](https://web.cs.ucdavis.edu/~doty/papers/ppsim.bib) ]

The core of the simulator uses a [batching algorithm](https://arxiv.org/abs/2005.03584) which gives significant asymptotic gains for protocols with relatively small reachable state sets. The package is designed to be run in a Python notebook, to concisely describe complex protocols, efficiently simulate their dynamics, and provide helpful visualization of the simulation.

## Table of contents

* [Installation](#installation)
* [First example protocol](#first-example-protocol)
* [Larger state protocol](#larger-state-protocol)
* [Protocol with Multiple Fields](#protocol-with-multiple-fields)
* [Simulating Chemical Reaction Networks (CRNs)](#simulating-chemical-reaction-networks-crns)

## Installation

The package can be installed with `pip` via


```python
pip install ppsim
```

The most important part of the package is the `Simulation` class, which is responsible for parsing a protocol, performing the simulation, and giving data about the simulation.


```python
from ppsim import Simulation
```

## First example protocol

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
<p>102 rows × 3 columns</p>
</div>




```python
p = sim.history.plot()
```


    
![png](https://github.com/UC-Davis-molecular-computing/ppsim/blob/main/README_files/README_12_0.png)
    


Without specifying an end time, `run` will run the simulation until the configuration is silent (all interactions are null). In this case, that will be when the protocol reaches a silent majority consensus configuration.


```python
sim.run()
p = sim.history.plot()
```

     Time: 21.000
    


    
![png](https://github.com/UC-Davis-molecular-computing/ppsim/blob/main/README_files/README_14_1.png)
    


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
    


    
![png](https://github.com/UC-Davis-molecular-computing/ppsim/blob/main/README_files/README_18_1.png)
    


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
<p>1497 rows × 2 columns</p>
</div>



This dataframe collected time from up to 100 trials for each population size n across a many orders of magnitude, limited by the budget of 30 seconds of wallclock time that we gave it.
We can now use the `seaborn` library to get a convincing plot of the data.


```python
import seaborn as sns
lp = sns.lineplot(x='n', y='time', data=df)
lp.set_xscale('log')
```


    
![png](https://github.com/UC-Davis-molecular-computing/ppsim/blob/main/README_files/README_22_0.png)
    


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

     Time: 14.800
    




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
      <td>450215</td>
      <td>1</td>
      <td>1</td>
      <td>20</td>
      <td>3</td>
      <td>1</td>
      <td>391</td>
      <td>134</td>
      <td>2</td>
      <td>8</td>
      <td>...</td>
      <td>9</td>
      <td>0</td>
      <td>125</td>
      <td>395</td>
      <td>0</td>
      <td>2</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>450243</td>
    </tr>
    <tr>
      <th>0.2</th>
      <td>401257</td>
      <td>11</td>
      <td>11</td>
      <td>229</td>
      <td>30</td>
      <td>14</td>
      <td>2125</td>
      <td>694</td>
      <td>18</td>
      <td>199</td>
      <td>...</td>
      <td>188</td>
      <td>26</td>
      <td>684</td>
      <td>2165</td>
      <td>11</td>
      <td>27</td>
      <td>176</td>
      <td>10</td>
      <td>7</td>
      <td>401337</td>
    </tr>
    <tr>
      <th>0.3</th>
      <td>354726</td>
      <td>46</td>
      <td>61</td>
      <td>715</td>
      <td>146</td>
      <td>70</td>
      <td>4818</td>
      <td>1643</td>
      <td>114</td>
      <td>721</td>
      <td>...</td>
      <td>753</td>
      <td>134</td>
      <td>1730</td>
      <td>5086</td>
      <td>75</td>
      <td>122</td>
      <td>720</td>
      <td>53</td>
      <td>33</td>
      <td>354312</td>
    </tr>
    <tr>
      <th>0.4</th>
      <td>310248</td>
      <td>106</td>
      <td>145</td>
      <td>1572</td>
      <td>360</td>
      <td>251</td>
      <td>8297</td>
      <td>2953</td>
      <td>340</td>
      <td>1720</td>
      <td>...</td>
      <td>1708</td>
      <td>399</td>
      <td>2926</td>
      <td>8523</td>
      <td>233</td>
      <td>327</td>
      <td>1653</td>
      <td>161</td>
      <td>116</td>
      <td>309999</td>
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
      <th>14.4</th>
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
      <th>14.5</th>
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
      <th>14.6</th>
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
      <th>14.7</th>
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
      <th>14.8</th>
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
<p>149 rows × 51 columns</p>
</div>



With a much larger number of states, the `history` dataframe is more unwieldly, so trying to directly call `history.plot()` would be very messy and not very useful. Instead, we will define a function that makes a barplot, using the data in a single row of `sim.history` to visualize the distribution at that recorded time step.


```python
from matplotlib import pyplot as plt
def plot_row(row):
    fig, ax = plt.subplots(figsize=(12,5))
    sim.history.iloc[row].plot(ax=ax, kind='bar', 
                              title=f'Discrete averaging at time {sim.history.index[row]:.2f}', 
                              xlabel='minute',
                              ylim=(0,n))
plot_row(0)
plot_row(30)
plot_row(-1)
```


    
![png](https://github.com/UC-Davis-molecular-computing/ppsim/blob/main/README_files/README_35_0.png)
    



    
![png](https://github.com/UC-Davis-molecular-computing/ppsim/blob/main/README_files/README_35_1.png)
    



    
![png](https://github.com/UC-Davis-molecular-computing/ppsim/blob/main/README_files/README_35_2.png)
    


The `ipywidgets` library gives a quick way to make a slider that lets us visualize the evolution of this distribution:


```python
import ipywidgets as widgets
bar = widgets.interact(plot_row, row = widgets.IntSlider(
    min=0, max=len(sim.history)-1, step=1, value=0, layout = widgets.Layout(width='100%')))
```

![gif](https://github.com/UC-Davis-molecular-computing/ppsim/blob/main/README_files/barplot1.gif)

It is recommended to use an interactive matplotlib backend, such as `ipympl`, which can be installed with `pip install ipympl` and then activated with the cell magic `%matplotlib widget`. The recommended environment to use for these notebooks is [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/). Unfortunately, these interactive backends are not supported with [Google Colab](https://colab.research.google.com/), so there does not seem to be an easy way to have access to interactive backends with something that can be run only in a browser without local installation.

The code with the slider above was designed to work in the non-interactive backend. The following cell shows how to accomplish the same thing with an interactive backend:


```python
# The following example uses the ipympl backend. It creates one figure and axis once and then modifies the axis directly with plot_row.
# If ipympl is installed, then uncommenting and running the following code will produce a slider that changes one single interactive figure object.

# %matplotlib widget
# def plot_row(row):
#     ax.clear()
#     sim.history.iloc[row].plot(ax=ax, kind='bar', 
#                               title=f'Discrete averaging at time {sim.history.index[row]:.2f}', 
#                               xlabel='minute',
#                               ylim=(0,n))
#     fig.canvas.draw()
    
# fig, ax = plt.subplots()
# bar = widgets.interact(plot_row, row = widgets.IntSlider(
#     min=0, max=len(sim.history)-1, step=1, value=0, layout = widgets.Layout(width='100%')))
```

Because the population average was exactly 25, the ultimate silent configuration will have every agent in state 50, but it will take a a very long time to reach, as we must wait for pairwise interactions between dwindling counts of states 24 and 26. We can check that this reaction is now the only possible non-null interaction.


```python
print(sim.enabled_reactions)
```

    24, 26  -->  25, 25
    

As a result, the probability of a non-null interaction will grow very small, upon which the simulator will switch to the Gillespie algorithm. This allows it to relatively quickly run all the way until silence, which we can confirm takes a very long amount of parallel time.


```python
# Setting history_interval to be a function of time t that shrinks, to not record too many configurations over a long time scale
sim.run(history_interval=lambda t: 10 ** len(str(int(t))) / 100)
```

     Time: 578983.800
    

To better visualize small count states, we add an option to change `yscale` from `linear` to `symlog`.


```python
def plot_row(row, yscale):
    fig, ax = plt.subplots(figsize=(12,5))
    sim.history.iloc[row].plot(ax=ax, kind='bar', 
                              title=f'Discrete averaging at time {sim.history.index[row]:.2f}', 
                              xlabel='minute',
                              ylim=(0,n))
    ax.set_yscale(yscale)
    
bar = widgets.interact(plot_row, 
                       row = widgets.IntSlider(min=0, max=len(sim.history)-1, step=1, value=0, layout = widgets.Layout(width='100%')),
                      yscale = ['linear','symlog'])
```

![gif](https://github.com/UC-Davis-molecular-computing/ppsim/blob/main/README_files/barplot2.gif)

## Protocol with Multiple Fields

For more complicated protocol, it is helpful to have the states be more complicated Python objects. A recommended method is to define an Agent [dataclass](https://docs.python.org/3/library/dataclasses.html) that includes various fields.

As a concrete example, we will use the protocol from [Simple and Efficient Leader Election](https://drops.dagstuhl.de/opus/volltexte/2018/8302/pdf/OASIcs-SOSA-2018-9.pdf). We start by translating the explicit description of an agents state into our Agent class.

![png](https://github.com/UC-Davis-molecular-computing/ppsim/blob/main/README_files/SimpleLeaderElection1.PNG)


```python
import dataclasses
from dataclasses import dataclass

# The parameter unsafe_hash=True makes the state hashable, as required, but still lets the transition code change the field values
# Note that ppsim will by default make safe copies of the agent states before applying the rule,
#  so it is safe to mutate the fields of an agent in the transition rule

@dataclass(unsafe_hash=True)
class Agent:
    role: str = 'contender'
    flip_bit: int = 0
    marker: int = 0
    phase: str = 'marking'
    counter: int = 0
```

![png](https://github.com/UC-Davis-molecular-computing/ppsim/blob/main/README_files/SimpleLeaderElection2.PNG)


```python
def leader_election(v: Agent, u: Agent, loglogn: int, Ulogn: int):
    # marking phase
    if v.phase == 'marking':
        if v.counter >= 3 * loglogn and u.flip_bit == 0:
            v.phase = 'tournament'
        else:
            v.counter += 1
        if v.counter == 4 * loglogn:
            v.marker = 1
            v.phase = 'tournament'
    
    if v.phase == 'tournament':
        if v.role == 'contender':
            if u.marker and v.counter <= Ulogn:
                v.counter += 1
            if v.counter < u.counter:
                v.role = 'minion'
            if u.role == 'contender' and v.counter == u.counter and v.flip_bit < u.flip_bit:
                v.role = 'minion'
        v.counter = max(v.counter, u.counter)
        
    v.flip_bit = 1 - v.flip_bit
    
    return v
```

The pseudocode was described in the following way:

![png](https://github.com/UC-Davis-molecular-computing/ppsim/blob/main/README_files/SimpleLeaderElection3.PNG)

We can implement this assumption by having our transition rule call the the `leader_election` function twice:


```python
def transition(v: Agent, u: Agent, loglogn: int, Ulogn: int):
    return leader_election(v, dataclasses.replace(u), loglogn, Ulogn), leader_election(u, dataclasses.replace(v), loglogn, Ulogn)
```

We can first check instantiate the protocol for various population sizes, to confirm that the number of reachable states is scaling like we expect.


```python
import numpy as np
ns = [int(n) for n in np.geomspace(10, 10 ** 8, 8)]
states = []
for n in ns:
    sim = Simulation({Agent(): n}, transition, loglogn=int(np.log2(np.log2(n))), Ulogn= u * int(np.log2(n)))
    states.append(len(sim.state_list))
plt.plot(ns, states)
plt.xscale('log')
plt.xlabel('population size n')
plt.ylabel('number of states')
plt.show()
```


    
![png](https://github.com/UC-Davis-molecular-computing/ppsim/blob/main/README_files/README_53_0.png)
    


Now we will simulate the rule for a population of one billion agents, and run it until it gets to one leader.


```python
n = 10 ** 9
sim = Simulation({Agent(): n}, transition, loglogn=int(np.log2(np.log2(n))), Ulogn= u * int(np.log2(n)))
def one_leader(config):
    leader_states = [state for state in config.keys() if state.role == 'contender']
    return len(leader_states) == 1 and config[leader_states[0]] == 1
sim.run(one_leader)
```

     Time: 67.253
    

Because there are hundreds of states, the full history dataframe is more complicated.


```python
sim.history
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>role</th>
      <th colspan="10" halign="left">contender</th>
      <th>...</th>
      <th colspan="10" halign="left">minion</th>
    </tr>
    <tr>
      <th>flip_bit</th>
      <th colspan="10" halign="left">0</th>
      <th>...</th>
      <th colspan="10" halign="left">1</th>
    </tr>
    <tr>
      <th>marker</th>
      <th colspan="10" halign="left">0</th>
      <th>...</th>
      <th colspan="10" halign="left">1</th>
    </tr>
    <tr>
      <th>phase</th>
      <th colspan="8" halign="left">marking</th>
      <th colspan="2" halign="left">tournament</th>
      <th>...</th>
      <th colspan="10" halign="left">tournament</th>
    </tr>
    <tr>
      <th>counter</th>
      <th>0</th>
      <th>2</th>
      <th>4</th>
      <th>6</th>
      <th>8</th>
      <th>10</th>
      <th>12</th>
      <th>14</th>
      <th>12</th>
      <th>13</th>
      <th>...</th>
      <th>50</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
      <th>58</th>
      <th>59</th>
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
      <th>0.000000</th>
      <td>1000000000</td>
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
      <th>1.000000</th>
      <td>135336837</td>
      <td>270661696</td>
      <td>90227329</td>
      <td>12028156</td>
      <td>859080</td>
      <td>38162</td>
      <td>1148</td>
      <td>6</td>
      <td>10</td>
      <td>8</td>
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
      <th>2.000000</th>
      <td>18312018</td>
      <td>146524448</td>
      <td>195383216</td>
      <td>104189030</td>
      <td>29773512</td>
      <td>5298122</td>
      <td>641440</td>
      <td>13908</td>
      <td>30153</td>
      <td>14976</td>
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
      <th>3.000000</th>
      <td>2478690</td>
      <td>44626126</td>
      <td>133867363</td>
      <td>160631023</td>
      <td>103263148</td>
      <td>41305948</td>
      <td>11269344</td>
      <td>556907</td>
      <td>1287669</td>
      <td>649759</td>
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
      <th>4.000000</th>
      <td>335736</td>
      <td>10736616</td>
      <td>57260444</td>
      <td>122138990</td>
      <td>139568370</td>
      <td>99261227</td>
      <td>48123925</td>
      <td>4232398</td>
      <td>10487317</td>
      <td>5527630</td>
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
      <th>64.000000</th>
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
      <th>65.000000</th>
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
      <th>66.000000</th>
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
      <th>67.000000</th>
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
      <th>67.252549</th>
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
<p>69 rows × 384 columns</p>
</div>



Because we defined a state as a dataclass `Agent`, which had fields, the columns of the `history` dataframe are a pandas [MultiIndex](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.MultiIndex.html).


```python
sim.history.columns
```




    MultiIndex([('contender', 0, 0,    'marking',  0),
                ('contender', 0, 0,    'marking',  2),
                ('contender', 0, 0,    'marking',  4),
                ('contender', 0, 0,    'marking',  6),
                ('contender', 0, 0,    'marking',  8),
                ('contender', 0, 0,    'marking', 10),
                ('contender', 0, 0,    'marking', 12),
                ('contender', 0, 0,    'marking', 14),
                ('contender', 0, 0, 'tournament', 12),
                ('contender', 0, 0, 'tournament', 13),
                ...
                (   'minion', 1, 1, 'tournament', 50),
                (   'minion', 1, 1, 'tournament', 51),
                (   'minion', 1, 1, 'tournament', 52),
                (   'minion', 1, 1, 'tournament', 53),
                (   'minion', 1, 1, 'tournament', 54),
                (   'minion', 1, 1, 'tournament', 55),
                (   'minion', 1, 1, 'tournament', 56),
                (   'minion', 1, 1, 'tournament', 57),
                (   'minion', 1, 1, 'tournament', 58),
                (   'minion', 1, 1, 'tournament', 59)],
               names=['role', 'flip_bit', 'marker', 'phase', 'counter'], length=384)



We can use the pandas [groupby](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html) function to conveniently look at the values of just one field. For a field whose name is the string `field`, then calling `sim.history.groupby(field, axis=1).sum()` gives the counts of values of just a single state. If we have a set of fields `field1, field2, ...` then calling `sim.history.groupby([field1, field2, ...], axis=1).sum()` will give the counts of values of just those fields.


```python
sim.history.groupby('role', axis=1).sum()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>role</th>
      <th>contender</th>
      <th>minion</th>
    </tr>
    <tr>
      <th>time (n interactions)</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.000000</th>
      <td>1000000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1.000000</th>
      <td>1000000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2.000000</th>
      <td>999999972</td>
      <td>28</td>
    </tr>
    <tr>
      <th>3.000000</th>
      <td>999969579</td>
      <td>30421</td>
    </tr>
    <tr>
      <th>4.000000</th>
      <td>998042414</td>
      <td>1957586</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>64.000000</th>
      <td>2</td>
      <td>999999998</td>
    </tr>
    <tr>
      <th>65.000000</th>
      <td>2</td>
      <td>999999998</td>
    </tr>
    <tr>
      <th>66.000000</th>
      <td>2</td>
      <td>999999998</td>
    </tr>
    <tr>
      <th>67.000000</th>
      <td>2</td>
      <td>999999998</td>
    </tr>
    <tr>
      <th>67.252549</th>
      <td>1</td>
      <td>999999999</td>
    </tr>
  </tbody>
</table>
<p>69 rows × 2 columns</p>
</div>



This lets us quickly plot the counts of leaders, to see how it decreases down to one leader, and the count in each phase, to see when the agents transition from the marking phase to the tournament phase.


```python
sim.history.groupby('role', axis=1).sum().plot()
plt.yscale('symlog')
plt.ylim(0, 2*n)
plt.show()
```


    
![png](https://github.com/UC-Davis-molecular-computing/ppsim/blob/main/README_files/README_63_0.png)
    



```python
sim.history.groupby('phase', axis=1).sum().plot()
plt.show()
```


    
![png](https://github.com/UC-Davis-molecular-computing/ppsim/blob/main/README_files/README_64_0.png)
    


For this protocol, a good understanding of why it is working comes from looking at the product of `role` and `counter` values. The way the protocol works is that contenders increase their counter values, which spread by epidemic among all minions, to eliminate other contenders with smaller counter values.

We will again try to visualize a single row of the dataframe that projects onto just the `role` and `counter` values. Calling `df.iloc[index]` gives us a [Series](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html)


```python
df = sim.history.groupby(['counter','role'], axis=1).sum()
df.iloc[10]
```




    counter  role     
    0        contender        2
    1        contender       48
    2        contender      441
    3        contender     2876
    4        contender    13600
                          ...  
    57       minion           0
    58       contender        0
             minion           0
    59       contender        0
             minion           0
    Name: 10.0, Length: 108, dtype: int64



Then calling `unstack()` on the series will give pull off the first field, and give us a dataframe that can immediately plotted as a multibar plot.


```python
df.iloc[10].unstack()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>role</th>
      <th>contender</th>
      <th>minion</th>
    </tr>
    <tr>
      <th>counter</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>441.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2876.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13600.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>55257.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>183276.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>523542.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1305762.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2908411.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>5815321.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>10577404.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>19519273.0</td>
      <td>144731.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>15633037.0</td>
      <td>2774308.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>12659059.0</td>
      <td>11179251.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>9794528.0</td>
      <td>30343520.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>31088600.0</td>
      <td>522306882.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>7599591.0</td>
      <td>292467764.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>951336.0</td>
      <td>21176355.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>82136.0</td>
      <td>856059.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>5498.0</td>
      <td>30677.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>277.0</td>
      <td>1087.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>10.0</td>
      <td>68.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>48</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>51</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>52</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>53</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>54</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>55</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>56</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>57</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>58</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>59</th>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[10].unstack().plot(kind='bar', figsize=(12,5))
plt.show()
```


    
![png](https://github.com/UC-Davis-molecular-computing/ppsim/blob/main/README_files/README_69_0.png)
    


Now we can define a function that creates one of these plots at an arbitrary row, to get a similar slider that lets us quickly visualize the evolution of the distributions.


```python
def plot_row(row, yscale):
    fig, ax = plt.subplots(figsize=(12,5))
    df.iloc[row].unstack().plot(ax=ax, kind='bar', 
                              ylim=(0,n))
    ax.set_yscale(yscale)

bar = widgets.interact(plot_row, 
                       row = widgets.IntSlider(min=0, max=len(sim.history)-1, step=1, value=0, layout = widgets.Layout(width='100%')),
                      yscale = ['linear','symlog'])
```

![gif](https://github.com/UC-Davis-molecular-computing/ppsim/blob/main/README_files/barplot3.gif)


## Simulating Chemical Reaction Networks (CRNs)

`ppsim` is able to simulate any Chemical Reaction Network that has only bimolecular (2-input, 2-output) and unimolecular (1-input, 1-output) reactions. There is a special syntax used to specify CRNs, such as

![png](https://github.com/UC-Davis-molecular-computing/ppsim/blob/main/README_files/CRN.PNG)



```python
from ppsim import species

a,b,c,d = species('A B C D')
crn = [(a+b | 2*c).k(0.5).r(4), (c >> d).k(5)]
```

First we define `species` objects `a,b,c,d`. We then create `crn`, a list of `reaction` objects, which are created by composing these species. Using the `>>` operator creates an irreversible (one-way) reaction, while using the `|` operator creates a reversible (two-way) reaction. A rate constant can be added with the method `reaction.k(...)`, and the reverse rate constant is added with the method `reaction.r(...)`. If not specified, rate constants are assumed to be 1.


```python
sim = Simulation({a: 2000, b:1000}, crn)
sim.run()
p = sim.history.plot()
```

     Time: 37.000
    

    
![png](https://github.com/UC-Davis-molecular-computing/ppsim/blob/main/README_files/README_75_1.png)
    


CRNs are normally modelled by Gillespie kinetics, which gives a continuous time Markov process. The unimolecular reaction `C ->(5) D` happens as a Poisson process with rate 5 &middot; #C. The forward bimolecular reaction `A+B ->(0.5) 2C` happens as a Poisson process with rate 0.5 &middot; (#A &middot; #B / v), and the reverse bimolecular reaction happens as a Poisson process with rate `4 * #B (\#B - 1) / (2*v)`, where `v` is the volume parameter.

When creating a `Simulation` with a list of `reaction` objects, `ppsim` will by default use this continuous time model.
By default, `ppsim` sets the volume `v` to be the population size `n`, which makes the time units independent of population size. In some models, this volume parameter is instead baked directly into the numerical rate constant. In this case, the volume should be set manually in the Simulation constructor, with `Simulation(..., volume = 1)`. In addition, if these numerical rate constants are specified in specific time units (such as per second), this can be specified with `Simulation(..., time_units='seconds')`, and then all times will appear with appropriate units.

For more details about the CRN model and how it is faithfully represented as a continuous time population protocol, see [this paper](https://arxiv.org/abs/2105.04702).

## More examples
See https://github.com/UC-Davis-molecular-computing/population-protocols-python-package/tree/main/examples/