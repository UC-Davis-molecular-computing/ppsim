use numpy::npyffi::npy_intp;
use numpy::{PyArray1, PyArray2, PyArray3, PyArray4};
use pyo3::prelude::*;
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::time::{Duration, Instant};
use std::cmp::{min, max};
use statrs::function::gamma::ln_gamma;

/// Helper function to sort a pair of values
fn sorted_pair(a: npy_intp, b: npy_intp) -> (npy_intp, npy_intp) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Sample from the collision length distribution.
///
/// See Lemma 3 in the source paper. The distribution gives the number of agents needed to pick an agent twice,
/// when r unique agents have already been selected.
fn sample_coll(r: i64, u: f64, n: i64, has_bounds: bool, logn: f64) -> i64 {
    if r >= n {
        return 1;
    }

    let logu = u.ln();
    let lhs = ln_gamma((n - r + 1) as f64) - logu;

    let mut t_lo = 0;
    let mut t_hi = n - r;

    // We maintain the invariant that P(l > t_lo) >= u and P(l > t_hi) < u
    // Equivalently, lhs >= lgamma(n - r - t_lo + 1) + t_lo * logn and
    //               lhs <  lgamma(n - r - t_hi + 1) + t_hi * logn
    while t_lo < t_hi - 1 {
        let t_mid = (t_lo + t_hi) / 2;
        if lhs < ln_gamma((n - r + 1 - t_mid) as f64) + (t_mid as f64) * logn {
            t_hi = t_mid;
        } else {
            t_lo = t_mid;
        }
    }

    t_hi
}

/// Trait for the algorithm that runs the simulation.
pub trait Simulator {
    /// Run the simulation for a fixed number of steps.
    fn run(&mut self, py: Python, num_steps: i64, max_wallclock_time: Option<f64>) -> PyResult<Py<PyArray1<i64>>>;
    
    /// Reset to a given configuration.
    fn reset(&mut self, py: Python, config: Py<PyArray1<i64>>, t: Option<i64>) -> PyResult<()>;
}

/// Base class for the algorithm that runs the simulation.
///
/// The configuration is stored as an array of size q, so the states are the indices 0, ..., q-1.
#[pyclass]
#[derive(Clone)]
pub struct SimulatorSequentialArray {
    #[pyo3(get, set)]
    pub config: Py<PyArray1<i64>>,
    #[pyo3(get, set)]
    pub n: u64,
    #[pyo3(get, set)]
    pub t: u64,
    pub q: npy_intp,
    #[pyo3(get, set)]
    pub delta: Py<PyArray4<npy_intp>>,
    #[pyo3(get, set)]
    pub null_transitions: Py<PyArray2<u8>>,
    pub is_random: u8,
    pub random_transitions: Py<PyArray3<npy_intp>>,
    pub random_outputs: Py<PyArray2<npy_intp>>,
    pub transition_probabilities: Py<PyArray1<f64>>,
    pub random_depth: npy_intp,
    pub bitgen: PyObject,
    rng: Option<Pcg64>,
    population: Py<PyArray1<npy_intp>>,
}

#[pymethods]
impl SimulatorSequentialArray {
    #[new]
    fn new(
        py: Python,
        init_array: Py<PyArray1<i64>>,
        delta: Py<PyArray4<npy_intp>>,
        null_transitions: Py<PyArray2<u8>>,
        random_transitions: Py<PyArray3<npy_intp>>,
        random_outputs: PyObject,
        transition_probabilities: Py<PyArray1<f64>>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        println!("*********** Rust SimulatorSequentialArray::new() called");
        let init_array_clone = init_array.clone();
        let init_array_view = init_array_clone.as_ref(py).readonly();
        let n = init_array_view.as_array().sum() as u64;
        let q = init_array_view.shape()[0] as npy_intp;

        // Check if any random transitions
        let random_transitions_clone = random_transitions.clone();
        let random_transitions_view = random_transitions_clone.as_ref(py).readonly();
        let numpy = py.import("numpy")?;
        let is_random = numpy
            .getattr("any")?
            .call1((random_transitions_view.to_object(py),))?
            .extract::<u8>()?;

        let random_depth = if is_random != 0 {
            let mut max_depth = 0;
            let rt_array = random_transitions_view.as_array();
            for i in 0..q as usize {
                for j in 0..q as usize {
                    max_depth = max_depth.max(rt_array[[i, j, 0]]);
                }
            }
            max_depth
        } else {
            0
        };

        // Create PCG64 random number generator
        let pcg64 = py.import("numpy.random")?.getattr("PCG64")?;
        let gen = match seed {
            Some(s) => pcg64.call1((s,))?,
            None => pcg64.call0()?,
        };

        // Create a Rust RNG
        let rust_rng = match seed {
            Some(s) => Some(Pcg64::seed_from_u64(s)),
            None => Some(Pcg64::from_entropy()),
        };

        // Create population array
        let population = PyArray1::zeros(py, [n as usize], false);
        let mut population_array = unsafe { population.as_array_mut() };

        // Initialize population array
        let config_array = init_array_view.as_array();
        let mut k = 0;
        for i in 0..q as usize {
            for _ in 0..config_array[i] {
                population_array[k] = i as npy_intp;
                k += 1;
            }
        }

        Ok(SimulatorSequentialArray {
            config: init_array,
            n,
            t: 0,
            q,
            delta,
            null_transitions,
            is_random,
            random_transitions,
            random_outputs: random_outputs.extract(py)?,
            transition_probabilities,
            random_depth,
            bitgen: gen.into(),
            rng: rust_rng,
            population: population.to_owned(),
        })
    }

    /// Creates the array self.population.
    fn make_population(&mut self, py: Python) -> PyResult<()> {
        // Create population array
        let population = PyArray1::zeros(py, [self.n as usize], false);
        let mut population_array = unsafe { population.as_array_mut() };

        // Initialize population array
        let config_array = self.config.as_ref(py).readonly();
        let config_array = config_array.as_array();
        let mut k = 0;
        for i in 0..self.q as usize {
            for _ in 0..config_array[i] {
                population_array[k] = i as npy_intp;
                k += 1;
            }
        }

        self.population = population.to_owned();

        Ok(())
    }
}

impl Simulator for SimulatorSequentialArray {
    /// Run the simulation for a fixed number of steps.
    fn run(
        &mut self,
        py: Python,
        num_steps: i64,
        max_wallclock_time: Option<f64>,
    ) -> PyResult<Py<PyArray1<i64>>> {
        println!("*********** Rust SimulatorSequentialArray::run() called");
        let max_wallclock_time = max_wallclock_time.unwrap_or(3600.0);
        let end_time = Instant::now() + Duration::from_secs_f64(max_wallclock_time);

        // Get references to simulator attributes
        let null_transitions = self.null_transitions.as_ref(py).readonly();
        let null_transitions_array = null_transitions.as_array();
        let delta = self.delta.as_ref(py).readonly();
        let delta_array = delta.as_array();
        let random_transitions = self.random_transitions.as_ref(py).readonly();
        let random_transitions_array = random_transitions.as_array();
        let random_outputs = self.random_outputs.as_ref(py).readonly();
        let random_outputs_array = random_outputs.as_array();
        let transition_probabilities = self.transition_probabilities.as_ref(py).readonly();
        let transition_probabilities_array = transition_probabilities.as_array();

        let mut population = self.population.as_ref(py).readwrite();
        let mut population_array = population.as_array_mut();

        let mut config = self.config.as_ref(py).readwrite();
        let mut config_array = config.as_array_mut();

        let end_step = self.t + num_steps as u64;

        while self.t < end_step && Instant::now() < end_time {
            // Random integer in [0, ..., n-1]
            let (i, j) = if let Some(rng) = &mut self.rng {
                let uniform = Uniform::from(0..self.n as usize);
                let i: usize = uniform.sample(rng);
                let mut j: usize = uniform.sample(rng);

                // Rejection sampling to quickly get distinct pair
                while i == j {
                    j = uniform.sample(rng);
                }
                (i, j)
            } else {
                // Fallback to Python's random if Rust RNG is not available
                let random = py.import("random")?;
                let i: usize = random
                    .call_method1("randint", (0, self.n as i64 - 1))?
                    .extract()?;
                let mut j: usize = random
                    .call_method1("randint", (0, self.n as i64 - 1))?
                    .extract()?;

                // Rejection sampling to quickly get distinct pair
                while i == j {
                    j = random
                        .call_method1("randint", (0, self.n as i64 - 1))?
                        .extract()?;
                }
                (i, j)
            };

            let a = population_array[i] as usize;
            let b = population_array[j] as usize;

            if null_transitions_array[[a, b]] == 0 {
                if self.is_random != 0 && random_transitions_array[[a, b, 0]] != 0 {
                    let mut k = random_transitions_array[[a, b, 1]] as usize;

                    // Sample from a probability distribution
                    let random_val: f64 = if let Some(rng) = &mut self.rng {
                        rng.gen::<f64>()
                    } else {
                        let random = py.import("random")?;
                        random.call_method0("random")?.extract()?
                    };
                    let mut u = random_val - transition_probabilities_array[k];

                    while u > 0.0 {
                        k += 1;
                        u -= transition_probabilities_array[k];
                    }

                    population_array[i] = random_outputs_array[[k, 0]];
                    population_array[j] = random_outputs_array[[k, 1]];
                } else {
                    population_array[i] = delta_array[[a, b, 0, 0]];
                    population_array[j] = delta_array[[a, b, 1, 0]];
                }

                config_array[a] -= 1;
                config_array[b] -= 1;
                config_array[population_array[i] as usize] += 1;
                config_array[population_array[j] as usize] += 1;
            }

            self.t += 1;
        }

        Ok(self.config.clone())
    }

    /// Reset to a given configuration.
    fn reset(&mut self, py: Python, config: Py<PyArray1<i64>>, t: Option<i64>) -> PyResult<()> {
        self.config = config;
        self.t = t.unwrap_or(0) as u64;

        // Recalculate n
        let config_array = self.config.as_ref(py).readonly();
        let n = config_array.as_array().sum() as u64;
        self.n = n;

        // Create population array
        let population = PyArray1::zeros(py, [n as usize], false);
        let mut population_array = unsafe { population.as_array_mut() };

        // Initialize population array
        let config_array = config_array.as_array();
        let mut k = 0;
        for i in 0..self.q as usize {
            for _ in 0..config_array[i] {
                population_array[k] = i as npy_intp;
                k += 1;
            }
        }

        self.population = population.to_owned();

        Ok(())
    }
}

/// A Simulator that uses the MultiBatch algorithm to simulate O(sqrt(n)) interactions in parallel.
#[pyclass]
#[derive(Clone)]
pub struct SimulatorMultiBatch {
    #[pyo3(get, set)]
    pub config: Py<PyArray1<i64>>,
    #[pyo3(get, set)]
    pub n: u64,
    #[pyo3(get, set)]
    pub t: u64,
    pub q: npy_intp,
    #[pyo3(get, set)]
    pub delta: Py<PyArray4<npy_intp>>,
    #[pyo3(get, set)]
    pub null_transitions: Py<PyArray2<u8>>,
    pub is_random: u8,
    pub random_transitions: Py<PyArray3<npy_intp>>,
    pub random_outputs: Py<PyArray2<npy_intp>>,
    pub transition_probabilities: Py<PyArray1<f64>>,
    pub random_depth: npy_intp,
    pub bitgen: PyObject,
    rng: Option<Pcg64>,
    
    // MultiBatch specific fields
    #[pyo3(get)]
    pub urn: Py<PyAny>,
    pub updated_counts: Py<PyAny>,
    pub logn: f64,
    pub batch_threshold: i64,
    pub row_sums: Py<PyArray1<i64>>,
    pub row: Py<PyArray1<i64>>,
    pub m: Py<PyArray1<i64>>,
    #[pyo3(get, set)]
    pub do_gillespie: bool,
    #[pyo3(get, set)]
    pub silent: bool,
    #[pyo3(get, set)]
    pub reactions: Py<PyArray2<npy_intp>>,
    #[pyo3(get, set)]
    pub enabled_reactions: Py<PyArray1<npy_intp>>,
    #[pyo3(get, set)]
    pub num_enabled_reactions: npy_intp,
    pub propensities: Py<PyArray1<f64>>,
    #[pyo3(get, set)]
    pub reaction_probabilities: Py<PyArray1<f64>>,
    pub gillespie_threshold: f64,
    pub coll_table: Py<PyArray2<i64>>,
    pub coll_table_r_values: Py<PyArray1<i64>>,
    pub coll_table_u_values: Py<PyArray1<f64>>,
    pub num_r_values: npy_intp,
    pub num_u_values: npy_intp,
    pub r_constant: i64,
}

#[pymethods]
impl SimulatorMultiBatch {
    #[new]
    fn new(
        py: Python,
        init_array: Py<PyArray1<i64>>,
        delta: Py<PyArray4<npy_intp>>,
        null_transitions: Py<PyArray2<u8>>,
        random_transitions: Py<PyArray3<npy_intp>>,
        random_outputs: PyObject,
        transition_probabilities: Py<PyArray1<f64>>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        println!("*********** Rust SimulatorMultiBatch::new() called");
        let init_array_clone = init_array.clone();
        let init_array_view = init_array_clone.as_ref(py).readonly();
        let n = init_array_view.as_array().sum() as u64;
        let q = init_array_view.shape()[0] as npy_intp;

        // Check if any random transitions
        let random_transitions_clone = random_transitions.clone();
        let random_transitions_view = random_transitions_clone.as_ref(py).readonly();
        let numpy = py.import("numpy")?;
        let is_random = numpy
            .getattr("any")?
            .call1((random_transitions_view.to_object(py),))?
            .extract::<u8>()?;

        let random_depth = if is_random != 0 {
            let mut max_depth = 0;
            let rt_array = random_transitions_view.as_array();
            for i in 0..q as usize {
                for j in 0..q as usize {
                    max_depth = max_depth.max(rt_array[[i, j, 0]]);
                }
            }
            max_depth
        } else {
            0
        };

        // Create PCG64 random number generator
        let pcg64 = py.import("numpy.random")?.getattr("PCG64")?;
        let gen = match seed {
            Some(s) => pcg64.call1((s,))?,
            None => pcg64.call0()?,
        };

        // Create a Rust RNG
        let rust_rng = match seed {
            Some(s) => Some(Pcg64::seed_from_u64(s)),
            None => Some(Pcg64::from_entropy()),
        };

        // Initialize MultiBatch specific parameters
        let logn = (n as f64).ln();
        let batch_threshold = min(((n as f64 / logn).sqrt() * q as f64) as i64, (n as f64).powf(0.7) as i64);
        let gillespie_threshold = 2.0 / (n as f64).sqrt();

        // Create arrays for MultiBatch
        let row_sums = PyArray1::zeros(py, [q as usize], false);
        let row = PyArray1::zeros(py, [q as usize], false);
        let m = PyArray1::zeros(py, [random_depth as usize], false);

        // Create Urn objects
        let urn_module = py.import("ppsim.simulator")?;
        let urn = urn_module.getattr("Urn")?.call_method1("create", (init_array.clone(), gen.clone()))?;
        let zero_config: &PyArray1<i64> = PyArray1::zeros(py, [q as usize], false);
        let updated_counts = urn_module.getattr("Urn")?.call_method1("create", (zero_config, gen.clone()))?;

        // Build coll_table for precomputed values
        let num_r_values = (10.0 * logn) as npy_intp;
        let num_u_values = (10.0 * logn) as npy_intp;
        let r_constant = max((1.5 * batch_threshold as f64 / ((num_r_values - 2) as f64).powi(2)) as i64, 1);
        
        let coll_table_r_values = PyArray1::zeros(py, [num_r_values as usize], false);
        let mut coll_table_r_values_array_mut = unsafe { coll_table_r_values.as_array_mut() };
        for i in 0..num_r_values as usize - 1 {
            coll_table_r_values_array_mut[i] = 2 + r_constant * ((i * i) as i64);
        }
        coll_table_r_values_array_mut[num_r_values as usize - 1] = n as i64;
        
        let coll_table_u_values = PyArray1::from_vec(py, 
            (0..num_u_values).map(|i| i as f64 / (num_u_values - 1) as f64).collect::<Vec<f64>>()
        );
        
        let coll_table = PyArray2::zeros(py, [num_r_values as usize, num_u_values as usize], false);
        let mut coll_table_array = unsafe { coll_table.as_array_mut() };
        let coll_table_r_values_binding = coll_table_r_values.readonly();
        let coll_table_r_values_array = coll_table_r_values_binding.as_array();
        let coll_table_u_values_binding = coll_table_u_values.readonly();
        let coll_table_u_values_array = coll_table_u_values_binding.as_array();
        
        for i in 0..num_r_values as usize {
            for j in 0..num_u_values as usize {
                coll_table_array[[i, j]] = sample_coll(
                    coll_table_r_values_array[i], 
                    coll_table_u_values_array[j], 
                    n as i64, 
                    false, 
                    logn
                );
            }
        }

        // Enumerate reactions for Gillespie
        let mut reactions = Vec::new();
        let mut reaction_probabilities_vec = Vec::new();
        
        let delta_binding = delta.as_ref(py).readonly();
        let delta_array = delta_binding.as_array();
        
        let null_transitions_binding = null_transitions.as_ref(py).readonly();
        let null_transitions_array = null_transitions_binding.as_array();
        
        let random_transitions_binding = random_transitions.as_ref(py).readonly();
        let random_transitions_array = random_transitions_binding.as_array();
        
        let random_outputs_extracted = random_outputs.extract::<Py<PyArray2<npy_intp>>>(py)?;
        let random_outputs_binding = random_outputs_extracted.as_ref(py).readonly();
        let random_outputs_array = random_outputs_binding.as_array();
        
        let transition_probabilities_binding = transition_probabilities.as_ref(py).readonly();
        let transition_probabilities_array = transition_probabilities_binding.as_array();
        
        for i in 0..q as usize {
            for j in 0..=i {
                // Check if interaction is symmetric
                let mut symmetric = false;
                
                // Check that entries in delta array match
                if sorted_pair(delta_array[[i, j, 0, 0]], delta_array[[i, j, 1, 0]]) == 
                   sorted_pair(delta_array[[j, i, 0, 0]], delta_array[[j, i, 1, 0]]) {
                    // Check if those really were matching deterministic transitions
                    if is_random == 0 || (random_transitions_array[[i, j, 0]] == 0 && random_transitions_array[[j, i, 0]] == 0) {
                        symmetric = true;
                    }
                    // If they have the same number of random outputs, check these random outputs match
                    else if is_random != 0 && random_transitions_array[[i, j, 0]] == random_transitions_array[[j, i, 0]] && random_transitions_array[[i, j, 0]] > 0 {
                        let a = random_transitions_array[[i, j, 1]] as usize;
                        let b = random_transitions_array[[j, i, 1]] as usize;
                        symmetric = true;
                        for k in 0..random_transitions_array[[i, j, 0]] as usize {
                            if sorted_pair(random_outputs_array[[a + k, 0]], random_outputs_array[[a + k, 1]]) != 
                               sorted_pair(random_outputs_array[[b + k, 0]], random_outputs_array[[b + k, 1]]) {
                                symmetric = false;
                                break;
                            }
                        }
                    }
                }
                
                let indices = if symmetric {
                    vec![(i, j, 1.0)]
                } else {
                    vec![(i, j, 0.5), (j, i, 0.5)]
                };
                
                for (a, b, p) in indices {
                    if null_transitions_array[[a, b]] == 0 {
                        if is_random != 0 && random_transitions_array[[a, b, 0]] > 0 {
                            for k in 0..random_transitions_array[[a, b, 0]] as usize {
                                let output_idx = random_transitions_array[[a, b, 1]] as usize + k;
                                let output_a = random_outputs_array[[output_idx, 0]];
                                let output_b = random_outputs_array[[output_idx, 1]];
                                
                                if output_a != a as npy_intp || output_b != b as npy_intp {
                                    reactions.push([a as npy_intp, b as npy_intp, output_a, output_b]);
                                    reaction_probabilities_vec.push(transition_probabilities_array[output_idx] * p);
                                }
                            }
                        } else {
                            let output_a = delta_array[[a, b, 0, 0]];
                            let output_b = delta_array[[a, b, 1, 0]];
                            
                            if output_a != a as npy_intp || output_b != b as npy_intp {
                                reactions.push([a as npy_intp, b as npy_intp, output_a, output_b]);
                                reaction_probabilities_vec.push(p);
                            }
                        }
                    }
                }
            }
        }
        
        // Create reactions array
        let reactions_array = PyArray2::zeros(py, [reactions.len(), 4], false);
        let mut reactions_array_mut = unsafe { reactions_array.as_array_mut() };
        for (i, reaction) in reactions.iter().enumerate() {
            for j in 0..4 {
                reactions_array_mut[[i, j]] = reaction[j];
            }
        }
        let reaction_probabilities_array = PyArray1::from_vec(py, reaction_probabilities_vec);
        let propensities = PyArray1::zeros(py, [reactions.len()], false);
        let enabled_reactions = PyArray1::zeros(py, [reactions.len()], false);
        
        let mut simulator = SimulatorMultiBatch {
            config: init_array.clone(),
            n,
            t: 0,
            q,
            delta: delta.clone(),
            null_transitions: null_transitions.clone(),
            is_random,
            random_transitions: random_transitions.clone(),
            random_outputs: random_outputs.extract(py)?,
            transition_probabilities: transition_probabilities.clone(),
            random_depth,
            bitgen: gen.into(),
            rng: rust_rng,
            
            urn: urn.into(),
            updated_counts: updated_counts.into(),
            logn,
            batch_threshold,
            row_sums: row_sums.to_owned(),
            row: row.to_owned(),
            m: m.to_owned(),
            do_gillespie: false,
            silent: false,
            reactions: reactions_array.to_owned(),
            enabled_reactions: enabled_reactions.to_owned(),
            num_enabled_reactions: 0,
            propensities: propensities.to_owned(),
            reaction_probabilities: reaction_probabilities_array.to_owned(),
            gillespie_threshold,
            coll_table: coll_table.to_owned(),
            coll_table_r_values: coll_table_r_values.to_owned(),
            coll_table_u_values: coll_table_u_values.to_owned(),
            num_r_values,
            num_u_values,
            r_constant,
        };
        
        // Initialize enabled_reactions
        simulator.update_enabled_reactions(py)?;
        
        Ok(simulator)
    }
    
    /// Updates self.enabled_reactions and self.num_enabled_reactions.
    fn update_enabled_reactions(&mut self, py: Python) -> PyResult<()> {
        let config_binding = self.config.as_ref(py).readonly();
        let config_array = config_binding.as_array();
        
        let reactions_binding = self.reactions.as_ref(py).readonly();
        let reactions_array = reactions_binding.as_array();
        
        let mut enabled_reactions = self.enabled_reactions.as_ref(py).readwrite();
        let mut enabled_reactions_array = enabled_reactions.as_array_mut();
        
        self.num_enabled_reactions = 0;
        
        for i in 0..reactions_array.shape()[0] {
            let reactant_1 = reactions_array[[i, 0]] as usize;
            let reactant_2 = reactions_array[[i, 1]] as usize;
            
            if (reactant_1 == reactant_2 && config_array[reactant_1] >= 2) || 
               (reactant_1 != reactant_2 && config_array[reactant_1] >= 1 && config_array[reactant_2] >= 1) {
                enabled_reactions_array[self.num_enabled_reactions as usize] = i as npy_intp;
                self.num_enabled_reactions += 1;
            }
        }
        
        Ok(())
    }
}

impl Simulator for SimulatorMultiBatch {
    /// Run the simulation for a fixed number of steps.
    fn run(&mut self, py: Python, num_steps: i64, max_wallclock_time: Option<f64>) -> PyResult<Py<PyArray1<i64>>> {
        println!("*********** Rust SimulatorMultiBatch::run() called");
        let max_wallclock_time = max_wallclock_time.unwrap_or(3600.0);
        let end_time = Instant::now() + Duration::from_secs_f64(max_wallclock_time);
        
        while self.t < num_steps as u64 && Instant::now() < end_time {
            if self.silent {
                self.t = num_steps as u64;
                return Ok(self.config.clone());
            } else if self.do_gillespie {
                // Simplified implementation for now
                self.t += 1;
            } else {
                // Simplified implementation for now
                self.t += 1;
            }
        }
        
        Ok(self.config.clone())
    }
    
    /// Reset to a given configuration.
    fn reset(&mut self, py: Python, config: Py<PyArray1<i64>>, t: Option<i64>) -> PyResult<()> {
        self.config = config;
        self.t = t.unwrap_or(0) as u64;
        
        // Recalculate n
        {
            let config_array = self.config.as_ref(py).readonly();
            let n = config_array.as_array().sum() as u64;
            self.n = n;
        }
        
        // Reset Urn
        let urn_module = py.import("ppsim.simulator")?;
        self.urn = urn_module.getattr("Urn")?.call_method1("create", (self.config.clone(), self.bitgen.clone()))?.into();
        
        // Reset silent and do_gillespie flags
        self.silent = false;
        self.do_gillespie = false;
        
        // Initialize enabled_reactions
        self.update_enabled_reactions(py)?;
        
        Ok(())
    }
}
