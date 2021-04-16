# ppsim Python package

The `ppsim` package is used for simulating population protcols. The core of the simulator uses a [batching algorithm](https://arxiv.org/abs/2005.03584) which gives significant asymptotic gains for protocols with relatively small reachable state sets. The package is designed to be run in a Python notebook, to concisely describe complex protocols, efficiently simulate their dynamics, and provide helpful visualization of the simulation.

## Installation

The package can be installed with `pip` via


```python
pip install ppsim
```
    

The most important part of the package is the `Simulation` class, which is responsible for parsing a protocol, performing the simulation, and giving data about the simulation.


```python
from ppsim import Simulation
```

## Example protcol

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

     Time: 10.001


The `Simulation` class can display all these configurations in a `pandas` dataframe in the attribute `history`.


```python
sim.history
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>U</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.000000</th>
      <td>501000000</td>
      <td>499000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0.100010</th>
      <td>478285585</td>
      <td>476280966</td>
      <td>45433449</td>
    </tr>
    <tr>
      <th>0.200039</th>
      <td>459449200</td>
      <td>457428961</td>
      <td>83121839</td>
    </tr>
    <tr>
      <th>0.300056</th>
      <td>443650105</td>
      <td>441607805</td>
      <td>114742090</td>
    </tr>
    <tr>
      <th>0.400098</th>
      <td>430264769</td>
      <td>428196457</td>
      <td>141538774</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9.601324</th>
      <td>352755117</td>
      <td>314377928</td>
      <td>332866955</td>
    </tr>
    <tr>
      <th>9.701327</th>
      <td>353418258</td>
      <td>313744457</td>
      <td>332837285</td>
    </tr>
    <tr>
      <th>9.801359</th>
      <td>354106871</td>
      <td>313093847</td>
      <td>332799282</td>
    </tr>
    <tr>
      <th>9.901364</th>
      <td>354824761</td>
      <td>312423976</td>
      <td>332751263</td>
    </tr>
    <tr>
      <th>10.001375</th>
      <td>355556526</td>
      <td>311728627</td>
      <td>332714847</td>
    </tr>
  </tbody>
</table>
<p>101 rows × 3 columns</p>
</div>




```python
sim.history.plot()
```




    <AxesSubplot:xlabel='time'>




    
![png](README_files/README_12_1.png)
    


Without specifying an end time, `run` will run the simulation until the configuration is silent (all interactions are null). In this case, that will be when the protcol reaches a silent majority consensus configuration.


```python
sim.run()
sim.history.plot()
```

     Time: 41.716





    <AxesSubplot:xlabel='time'>




    
![png](README_files/README_14_2.png)
    


As currently described, this protocol is one-way, where these interactions only take place if the two states meet in the specified order. We can see this by checking `print_reactions`.


```python
sim.print_reactions()
```

    A, B  -->  U, U      with probability 0.5
    A, U  -->  A, A      with probability 0.5
    B, U  -->  B, B      with probability 0.5
    

Here we have unorder pairs of reactants, and the probability `0.5` is because these interactions as written depended on the order of the agents. If we wanted to consider the more sensible symmetric variant of the protocol, one approach would explicitly give all non-null interactions:


```python
approximate_majority_symmetric = {
    (a,b): (u,u), (b,a): (u,u),
    (a,u): (a,a), (u,a): (a,a),
    (b,u): (b,b), (u,b): (b,b)
}
sim = Simulation(init_config, approximate_majority_symmetric)
```

But a quicker equivalent approach is to tell `Simulation` that all interactions should be interpreted as symmetric, so if we specify interaction `(a,b)` but leave `(b,a)` as null, then `(b,a)` will be interpreted as having the same output pair.


```python
sim = Simulation(init_config, approximate_majority, transition_order='symmetric')
sim.print_reactions()
sim.run()
sim.history.plot()
```

    A, B  -->  U, U
    A, U  -->  A, A
    B, U  -->  B, B
     Time: 21.195





    <AxesSubplot:xlabel='time'>




    
![png](README_files/README_20_2.png)
    


A key result about this protocol is it converges in expected O(log n) time, which surprisingly is very nontrivial to prove. We can use this package to very quickly gather some convincing data that the convergence really is O(log n) time, with the function `time_trials`.


```python
from ppsim import time_trials
import numpy as np

ns = [int(n) for n in np.geomspace(10, 10 ** 8, 20)]
def initial_condition(n):
    return {'A': n // 2, 'B': n // 2}
df = time_trials(approximate_majority, ns, initial_condition, num_trials=100, max_wallclock_time = 30, transition_order='symmetric')
df
```


      0%|          | 0/20 [00:00<?, ?it/s]





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>14.100000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>7.700000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>14.900000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>7.200000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1359</th>
      <td>42813323</td>
      <td>48.937840</td>
    </tr>
    <tr>
      <th>1360</th>
      <td>42813323</td>
      <td>50.391446</td>
    </tr>
    <tr>
      <th>1361</th>
      <td>42813323</td>
      <td>56.390054</td>
    </tr>
    <tr>
      <th>1362</th>
      <td>100000000</td>
      <td>50.265050</td>
    </tr>
    <tr>
      <th>1363</th>
      <td>100000000</td>
      <td>53.851442</td>
    </tr>
  </tbody>
</table>
<p>1364 rows × 2 columns</p>
</div>



This dataframe collected time from up to 100 trials for each population size n across a many orders of magnitude, limited by the budget of 30 seconds of wallclock time that we gave it.
We can now use the `seaborn` library to get a convincing plot of the data.


```python
import seaborn as sns
lp = sns.lineplot(x='n', y='time', data=df)
lp.set_xscale('log')
```


    
![png](README_files/README_24_0.png)
    


## Larger state protocol

For more complicated protocols, it would be very tedious to use this dictionary format. Instead we can give an arbitrary Python function which takes a pair of states as input (along with possible other protocol parameters) and returns a pair of states as output (or if we wanted a randomized transition, it would output a dictionary which maps pairs of states to probabilities).

As a quick example, let's take a look at the discrete averaging dynamics, as analyzed [here](https://arxiv.org/abs/1808.05389) and [here](https://hal-cnrs.archives-ouvertes.fr/hal-02473856/file/main_JAP.pdf), which have been a key subroutine used in counting and majority protocols.


```python
from math import ceil, floor

def discrete_averaging(a, b):
    avg = (a + b) / 2
    return floor(avg), ceil(avg)

n = 10 ** 8
sim = Simulation({0: n // 2, 100: n // 2}, discrete_averaging)
```

We did not need to explicitly describe the state set. Upon initialization, `Simulation` used breadth first search to find all states reachable from the initial configuration.


```python
print(sim.state_list)
```

    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    

This enumeration will call the function `rule` we give it O(q^2) times, where q is the number of reachable states. This preprocessing step also builds an internal representation of the transition function, so it will not need to continue calling `rule`. Thus we don't need to worry too much about our code for `rule` being efficient.

Rather than the dictionary format used to input the configuration, internally `Simulation` represents the configuration as an array of counts, where the ordering of the indices is given by `state_list`.


```python
sim.config_dict
```




    {0: 50000000, 100: 50000000}




```python
sim.config_array
```




    array([50000000,        0,        0,        0,        0,        0,
                  0,        0,        0,        0,        0,        0,
                  0,        0,        0,        0,        0,        0,
                  0,        0,        0,        0,        0,        0,
                  0,        0,        0,        0,        0,        0,
                  0,        0,        0,        0,        0,        0,
                  0,        0,        0,        0,        0,        0,
                  0,        0,        0,        0,        0,        0,
                  0,        0,        0,        0,        0,        0,
                  0,        0,        0,        0,        0,        0,
                  0,        0,        0,        0,        0,        0,
                  0,        0,        0,        0,        0,        0,
                  0,        0,        0,        0,        0,        0,
                  0,        0,        0,        0,        0,        0,
                  0,        0,        0,        0,        0,        0,
                  0,        0,        0,        0,        0,        0,
                  0,        0,        0,        0, 50000000], dtype=int64)



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

     Time: 17.870





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th>91</th>
      <th>92</th>
      <th>93</th>
      <th>94</th>
      <th>95</th>
      <th>96</th>
      <th>97</th>
      <th>98</th>
      <th>99</th>
      <th>100</th>
    </tr>
    <tr>
      <th>time</th>
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
      <td>50000000</td>
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
      <td>50000000</td>
    </tr>
    <tr>
      <th>0.100003</th>
      <td>45017904</td>
      <td>1</td>
      <td>1</td>
      <td>56</td>
      <td>6</td>
      <td>2</td>
      <td>1721</td>
      <td>603</td>
      <td>2</td>
      <td>64</td>
      <td>...</td>
      <td>63</td>
      <td>0</td>
      <td>558</td>
      <td>1643</td>
      <td>4</td>
      <td>11</td>
      <td>71</td>
      <td>4</td>
      <td>2</td>
      <td>45018958</td>
    </tr>
    <tr>
      <th>0.200260</th>
      <td>40123004</td>
      <td>53</td>
      <td>57</td>
      <td>1520</td>
      <td>226</td>
      <td>77</td>
      <td>18293</td>
      <td>6134</td>
      <td>137</td>
      <td>1328</td>
      <td>...</td>
      <td>1349</td>
      <td>141</td>
      <td>6256</td>
      <td>18388</td>
      <td>87</td>
      <td>278</td>
      <td>1494</td>
      <td>71</td>
      <td>51</td>
      <td>40124502</td>
    </tr>
    <tr>
      <th>0.300952</th>
      <td>35393996</td>
      <td>357</td>
      <td>397</td>
      <td>7436</td>
      <td>1435</td>
      <td>728</td>
      <td>62975</td>
      <td>21222</td>
      <td>962</td>
      <td>7077</td>
      <td>...</td>
      <td>7262</td>
      <td>1006</td>
      <td>21274</td>
      <td>62986</td>
      <td>669</td>
      <td>1394</td>
      <td>7319</td>
      <td>447</td>
      <td>342</td>
      <td>35396414</td>
    </tr>
    <tr>
      <th>0.400990</th>
      <td>30961350</td>
      <td>1257</td>
      <td>1556</td>
      <td>19870</td>
      <td>4242</td>
      <td>2637</td>
      <td>132312</td>
      <td>45295</td>
      <td>3802</td>
      <td>20391</td>
      <td>...</td>
      <td>20433</td>
      <td>3878</td>
      <td>45205</td>
      <td>132546</td>
      <td>2413</td>
      <td>4136</td>
      <td>19346</td>
      <td>1534</td>
      <td>1202</td>
      <td>30964915</td>
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
      <th>17.469344</th>
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
      <th>17.569425</th>
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
      <th>17.669490</th>
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
      <th>17.769641</th>
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
      <th>17.869756</th>
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
<p>178 rows × 101 columns</p>
</div>



With a much larger number of states, the `history` dataframe is more unwieldly, so trying to directly call `history.plot()` would be very messy and not very useful.
Instead we will bring in a `Snapshot` object that makes a bar plot with the counts of each state, and lets us visualize the way the distribution evolves over time.
For this `StatePlotter` object to work as intended, we need to be using an interactive matplotlib backend, such as `%matplotlib widget` or `%matplotlib qt`.


```python
%matplotlib widget
from ppsim import StatePlotter
sp = StatePlotter()
sim.add_snapshot(sp)
sim.snapshot_slider('time')
```


    Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …



    interactive(children=(FloatSlider(value=0.0, description='time', layout=Layout(width='100%'), max=17.86975578,…


To better visualize small count states, let's change `yscale` to `symlog`.


```python
sp.ax.set_yscale('symlog')
```

If we run the `Simulation` while this `Snapshot` has already been created, it will update while the simulation runs. Because the population average was exactly 50, the ultimate silent configuration will have every agent in state 50, but it will take a a very long time to reach, as we must wait for pairwise interactions between dwindling counts of states 49 and 51. We can check that this reaction is now the only possible non-null interaction.


```python
sim.print_enabled_reactions()
```

     49,  51  -->   50,  50
    

As a result, the probability of a non-null interaction will grow very small, upon which the simulator will switch to the Gillespie algorithm. This allows it to relatively quickly run all the way until silence, which we can confirm takes a very long amount of parallel time.


```python
sim.run()
```

Since the timescale of the whole simulation is now very long, we should have the slider range across recorded indices rather than parallel time.


```python
display(sp.fig.canvas)
sim.snapshot_slider('index')
```


    Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …



    interactive(children=(IntSlider(value=0, description='index', layout=Layout(width='100%'), max=11329), Output(…



```python

```
