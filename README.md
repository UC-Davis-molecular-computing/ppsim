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


The `Simulation` class can display all these configurations in a `pandas` dataframe in the attribute `history`.


```python
sim.history
```



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


    
![png](README_files/README_12_1.png)
    


Without specifying an end time, `run` will run the simulation until the configuration is silent (all interactions are null). In this case, that will be when the protcol reaches a silent majority consensus configuration.


```python
sim.run()
sim.history.plot()
```

    
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
<p>1364 rows x 2 columns</p>
</div>



This dataframe collected time from up to 100 trials for each population size n across a many orders of magnitude, limited by the budget of 30 seconds of wallclock time that we gave it.
We can now use the `seaborn` library to get a convincing plot of the data.


```python
import seaborn as sns
lp = sns.lineplot(x='n', y='time', data=df)
lp.set_xscale('log')
```


    
![png](README_files/README_24_0.png)
    
